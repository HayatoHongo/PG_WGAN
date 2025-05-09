import time
import torch
import streamlit as st
import matplotlib.pyplot as plt

# --- 設定 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_STEPS = 100
DELTA_T = 0.1
SCALE = 1000.0  # m → normalized units
PRETRAIN_G_PATH = "generator_finetuned_physics_stage3.pth"  # モデルファイルは同ディレクトリに配置してください

# --- モデル定義 ---
class Generator(torch.nn.Module):
    def __init__(self, noise_dim=2, label_dim=2, output_dim=NUM_STEPS*2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(noise_dim + label_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, noise, theta):
        flat = self.net(torch.cat([noise, theta], dim=1))
        return flat.view(-1, NUM_STEPS, 2)

# --- モデル読み込みキャッシュ ---
@st.cache(allow_output_mutation=True)
def load_model():
    g = Generator().to(DEVICE)
    g.load_state_dict(torch.load(PRETRAIN_G_PATH, map_location=DEVICE))
    g.eval()
    return g

# --- 軌道計算関数 ---
def compute_trajectories(v0_value, phi_value):
    # 初期設定
    phi = torch.tensor([phi_value * torch.pi/180.], device=DEVICE)
    v0 = torch.tensor([v0_value], device=DEVICE)
    theta = torch.stack([v0 * torch.cos(phi), v0 * torch.sin(phi)], dim=1)
    noise = torch.randn(1, 2, device=DEVICE)
    x0 = torch.zeros(1, 2, device=DEVICE)
    g = torch.tensor([[0.0, -9.8]], device=DEVICE)

    # 予測
    G = load_model()
    with torch.no_grad():
        fake_norm = G(noise, theta)
    fake_seq = fake_norm * SCALE  # m単位

    # 解析解
    t = torch.arange(1, NUM_STEPS+1, device=DEVICE).view(1, NUM_STEPS, 1) * DELTA_T
    real_seq = x0 + theta.view(-1, 1, 2) * t + 0.5 * g.view(1, 1, 2) * t**2

    return real_seq.squeeze(0).cpu().numpy(), fake_seq.squeeze(0).cpu().numpy()

# --- Streamlit UI ---
st.title("Trajectory Comparison: Physical vs DNN Generated")

# ユーザー入力スライダー
v0 = st.slider("初速度 v₀ (m/s)", min_value=0.0, max_value=100.0, value=50.0)
phi = st.slider("投射角度 φ (°)", min_value=0.0, max_value=90.0, value=45.0)

# 実行ボタン
if st.button("Run Animation"):
    real_traj, fake_traj = compute_trajectories(v0, phi)
    placeholder = st.empty()

    # アニメーション表示
    for i in range(len(real_traj)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(real_traj[:i, 0], real_traj[:i, 1], '-o', markersize=4, label='Physical Model')
        ax.plot(fake_traj[:i, 0], fake_traj[:i, 1], '-x', markersize=4, label='DNN Generated')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Trajectory (v₀={v0:.1f} m/s, φ={phi:.1f}°)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        placeholder.pyplot(fig)
        time.sleep(0.03)
    # 最終フレームを保持
    placeholder.pyplot(fig)
