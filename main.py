import streamlit as st
import torch
import matplotlib.pyplot as plt

# --- 設定 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_STEPS = 100
DELTA_T = 0.1
PRETRAIN_G_PATH = "generator_finetuned_physics_stage3.pth"
SCALE = 1000.0  # m → normalized units

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

# --- 物理残差関数 ---
def physics_residual(seq_m, x0, g, v0):
    t = torch.arange(1, NUM_STEPS+1, device=seq_m.device).view(1, NUM_STEPS, 1) * DELTA_T
    x0_e = x0.view(1,1,2)
    v0_e = v0.view(-1,1,2)
    g_e  = g.view(1,1,2)
    expected = x0_e + v0_e * t + 0.5 * g_e * t**2
    residuals = torch.norm(seq_m - expected, dim=2)
    return residuals.mean(dim=1)

# --- モデルロード ---
@st.cache_resource
def load_model(path):
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

G = load_model(PRETRAIN_G_PATH)

# --- Streamlit UI ---
st.title("Projectile Trajectory Comparison")

v0 = st.slider("初速 v₀ (m/s)", 0.0, 100.0, 60.0)
phi_deg = st.slider("投射角度 φ (度)", 0.0, 90.0, 20.0)

if st.button("軌道を生成"):
    # 準備
    phi = torch.tensor([phi_deg * torch.pi/180.], device=DEVICE)
    theta = torch.stack([v0 * torch.cos(phi), v0 * torch.sin(phi)], dim=1)
    noise = torch.randn(1, 2, device=DEVICE)
    x0 = torch.zeros(1,2, device=DEVICE)
    g = torch.tensor([[0.0, -9.8]], device=DEVICE)

    # DNN推論
    with torch.no_grad():
        fake_norm = G(noise, theta)
        fake_seq = fake_norm * SCALE

    # 解析解計算
    t = torch.arange(1, NUM_STEPS+1, device=DEVICE).view(1,NUM_STEPS,1) * DELTA_T
    real_seq = x0 + theta.view(-1,1,2) * t + 0.5 * g.view(1,1,2) * t**2

    # 残差表示
    res = physics_residual(fake_seq, x0, g, theta)
    st.write(f"物理残差 (平均L2誤差): {res.item():.4f} m")

    # プロット
    traj_fake = fake_seq.squeeze(0).cpu().numpy()
    traj_real = real_seq.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(traj_real[:,0], traj_real[:,1], '-o', markersize=4, label='Physical Model')
    ax.plot(traj_fake[:,0], traj_fake[:,1], '-x', markersize=4, label='DNN Prediction')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'φ={phi_deg}°, v₀={v0} m/s')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
