import sys
import os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import json
import os
from typing import Optional, Tuple, List, Union
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import httpx
import fire

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib, StockData
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC, EXPLAIN_WITH_TEXT_DESC_AND_COUNTEREXAMPLES
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction

stock_data_path = f"/data/cn_data"


# 添加颜色定义
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_llm_message(role: str, content: str):
    """打印LLM对话消息"""
    if role.lower() == "system":
        color = Colors.HEADER
    elif role.lower() == "user":
        color = Colors.OKBLUE
    elif role.lower() == "assistant":
        color = Colors.OKGREEN
    else:
        color = Colors.WARNING

    print(f"{color}[{role}]{Colors.ENDC}: {content}")


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))

    # 创建一个自定义的OpenAIClient类来添加输出
    class VerboseOpenAIClient(OpenAIClient):
        def chat_complete(self, content: str) -> str:
            print_llm_message("user", content)
            try:
                result = super().chat_complete(content)
                print_llm_message("assistant", result)
                return result
            except Exception as e:
                print_llm_message("system", f"Error during chat completion: {str(e)}")
                raise

    # 创建一个简单的httpx客户端，避免任何可能的参数冲突
    try:
        # 尝试创建不带任何额外参数的httpx客户端
        http_client = httpx.Client()
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-47d9931d19af499fa43171a74cdbb494",
            http_client=http_client
        )
    except Exception as e:
        print_llm_message("system", f"Failed to create OpenAI client with custom http_client: {e}")
        # 如果自定义http_client失败，尝试不使用http_client参数
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-47d9931d19af499fa43171a74cdbb494"
        )

    return VerboseOpenAIClient(
        client=client,
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC_AND_COUNTEREXAMPLES,  # Use the version with counterexamples
            logger=logger
        ),
        model="deepseek-chat",  # DeepSeek的模型名称
        max_tokens=4096  # DeepSeek的最大token限制
    )


class CustomCallback(BaseCallback):
    def __init__(
            self,
            save_path: str,
            test_calculators: List[QLibStockDataCalculator],
            verbose: int = 0,
            chat_session: Optional[InterativeSession] = None,
            llm_every_n_steps: int = 25_000,
            drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n

        # 添加绘图相关的属性
        self.ic_history: List[Tuple[int, float]] = []
        self.rank_ic_history: List[Tuple[int, float]] = []
        self.pool_size_history: List[Tuple[int, int]] = []
        self.plot_every_n_steps = 1000  # 每1000步绘制一次
        self.last_plot = 0

        print_llm_message("system", f"Initialized CustomCallback with llm_every_n_steps={llm_every_n_steps}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)

        # 记录历史数据用于绘图
        self.ic_history.append((self.num_timesteps, ic_test_mean))
        self.rank_ic_history.append((self.num_timesteps, rank_ic_test_mean))
        self.pool_size_history.append((self.num_timesteps, self.pool.size))

        # 定期绘制图表
        if self.num_timesteps - self.last_plot >= self.plot_every_n_steps:
            self._plot_training_curves()
            self.last_plot = self.num_timesteps

        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)  # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            state = self.pool.state
            # 转换Expression对象为字符串
            state['exprs'] = [str(expr) for expr in state['exprs']]
            json.dump(state, f)

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1

        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        print_llm_message("system",
                          f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
                          f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            print_llm_message("system", f"Remaining pool size: {remain_n}")
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            print_llm_message("system", f"LLM invocation failed due to {type(e)}: {str(e)}")
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    def _plot_training_curves(self):
        """绘制训练过程的IC曲线"""
        if len(self.ic_history) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Step {self.num_timesteps}', fontsize=16)

        # 提取数据
        steps = [x[0] for x in self.ic_history]
        ic_values = [x[1] for x in self.ic_history]
        rank_ic_values = [x[1] for x in self.rank_ic_history]
        pool_sizes = [x[1] for x in self.pool_size_history]

        # 1. IC曲线
        axes[0, 0].plot(steps, ic_values, 'b-', linewidth=2, label='IC')
        axes[0, 0].set_title('Information Coefficient (IC)')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('IC Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 2. Rank IC曲线
        axes[0, 1].plot(steps, rank_ic_values, 'r-', linewidth=2, label='Rank IC')
        axes[0, 1].set_title('Rank Information Coefficient')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Rank IC Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # 3. Pool Size变化
        axes[1, 0].plot(steps, pool_sizes, 'g-', linewidth=2, label='Pool Size')
        axes[1, 0].set_title('Alpha Pool Size')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Pool Size')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # 4. IC和Rank IC对比
        axes[1, 1].plot(steps, ic_values, 'b-', linewidth=2, label='IC')
        axes[1, 1].plot(steps, rank_ic_values, 'r-', linewidth=2, label='Rank IC')
        axes[1, 1].set_title('IC vs Rank IC Comparison')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(self.save_path, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印当前状态
        print(f"\n{Colors.OKGREEN}[Training Progress]{Colors.ENDC}")
        print(f"Step: {self.num_timesteps:,}")
        print(f"Current IC: {ic_values[-1]:.4f}")
        print(f"Current Rank IC: {rank_ic_values[-1]:.4f}")
        print(f"Pool Size: {pool_sizes[-1]}")
        print(f"Best IC so far: {max(ic_values):.4f}")
        print(f"Chart saved to: {plot_path}")
        print("-" * 50)

    def on_training_end(self):
        """训练结束时绘制最终图表"""
        if len(self.ic_history) > 0:
            self._plot_training_curves()
            # 保存训练历史数据
            history_data = {
                'ic_history': self.ic_history,
                'rank_ic_history': self.rank_ic_history,
                'pool_size_history': self.pool_size_history
            }
            history_path = os.path.join(self.save_path, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"{Colors.OKGREEN}[Training Complete]{Colors.ENDC}")
            print(f"Training history saved to: {history_path}")
            print(f"Final training curves saved to: {os.path.join(self.save_path, 'training_curves.png')}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert (isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def run_single_experiment(
        seed: int = 0,
        instruments: str = "csi300",
        pool_capacity: int = 10,
        steps: int = 200_000,
        alphagpt_init: bool = False,
        use_llm: bool = False,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5,
        llm_replace_n: int = 3
):
    reseed_everything(seed)
    initialize_qlib(stock_data_path)

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")
    # 生成实验的存储路径
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # tag = "rlv2" if llm_add_subexpr == 0 else f"afs{llm_add_subexpr}aar1-5"
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("available device:", device)

    # 计算目标变量target，即 未来 20 天的收益率
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        ("2022-01-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2023-07-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30")
    ]
    datasets = [get_dataset(*s) for s in segments]
    print('datastes loaded')
    # 计算segments对应的target（收益率）
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        n_steps=512,
        tensorboard_log="./out/tensorboard",
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )

    # 训练结束后绘制最终图表
    checkpoint_callback.on_training_end()


def main(
        random_seeds: Union[int, Tuple[int]] = 0,
        pool_capacity: int = 20,
        instruments: str = "csi300",
        alphagpt_init: bool = False,
        use_llm: bool = False,
        drop_rl_n: int = 10,
        steps: Optional[int] = None,
        llm_every_n_steps: int = 25000
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps
        )


if __name__ == '__main__':
    fire.Fire(main)
