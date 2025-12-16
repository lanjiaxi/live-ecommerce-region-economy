# 导入基础包+克里金法依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from pykrige.ok import OrdinaryKriging
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 共线性检验

# -------------------------- 中文配置 --------------------------
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 1. 加载扩展数据集 --------------------------
data = pd.read_csv("./input/economic.csv", encoding="utf-8")
data = data.dropna()

# 打印数据基本信息
print("扩展数据集基本信息：")
print(f"样本量：{len(data)} 行，省份数：{len(data['province'].unique())} 个")
print(data[["province", "year", "live_ecom", "gdp_growth"]].head())

# -------------------------- 2. 共线性检验（新增） --------------------------
print("\n===== 共线性检验（VIF值） =====")
X_vif = data[["live_ecom", "express", "internet"]]
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data["变量"] = X_vif.columns
vif_data["VIF值"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)  # VIF>10表示严重共线性

# -------------------------- 3. 基础相关性分析 --------------------------
print("\n===== 基础皮尔逊相关性分析 =====\n")
# 直播电商 vs GDP增速（分年份）
for year in [2021, 2022]:
    year_data = data[data["year"] == year]
    corr, p = pearsonr(year_data["live_ecom"], year_data["gdp_growth"])
    print(f"{year}年：直播电商 vs GDP增速 → 相关系数={corr:.3f}，p值={p:.3f}（{'显著' if p<0.05 else '不显著'}）")

# 整体相关性
corr_total, p_total = pearsonr(data["live_ecom"], data["gdp_growth"])
print(f"整体（2021-2022）：直播电商 vs GDP增速 → 相关系数={corr_total:.3f}，p值={p_total:.3f}（{'显著' if p_total<0.05 else '不显著'}）")

# -------------------------- 4. 克里金法空间相关性（2022年截面） --------------------------
print("\n===== 克里金法空间相关性分析（2022年）=====")
data_2022 = data[data["year"] == 2022]
lon = data_2022["lon"].values
lat = data_2022["lat"].values
z_ecom = data_2022["live_ecom"].values
z_gdp = data_2022["gdp_growth"].values

# 克里金插值
ok_ecom = OrdinaryKriging(lon, lat, z_ecom, variogram_model="spherical", verbose=False, enable_plotting=False)
ok_gdp = OrdinaryKriging(lon, lat, z_gdp, variogram_model="spherical", verbose=False, enable_plotting=False)

grid_lon = np.linspace(100, 125, 50)
grid_lat = np.linspace(20, 40, 50)
z_ecom_interp, _ = ok_ecom.execute("grid", grid_lon, grid_lat)
z_gdp_interp, _ = ok_gdp.execute("grid", grid_lon, grid_lat)

# 空间相关性
ecom_flat = z_ecom_interp.flatten()
gdp_flat = z_gdp_interp.flatten()
valid_idx = ~(np.isnan(ecom_flat) | np.isnan(gdp_flat))
spatial_corr, spatial_p = pearsonr(ecom_flat[valid_idx], gdp_flat[valid_idx])
print(f"2022年空间插值相关性：相关系数={spatial_corr:.3f}，p值={spatial_p:.3f}（{'显著' if spatial_p<0.05 else '不显著'}）")

# -------------------------- 5. 优化版回归分析（解决共线性） --------------------------
print("\n===== 回归分析结果 =====\n")
# 方案1：剔除高共线性变量（保留核心解释变量+1个控制变量）
X_opt1 = data[["live_ecom", "internet"]]  # 剔除快递业务量（与直播电商共线性最高）
X_opt1 = sm.add_constant(X_opt1)
y = data["gdp_growth"]
model_opt1 = sm.OLS(y, X_opt1).fit()
print("【方案1：剔除高共线性变量】")
print(f"R² = {model_opt1.rsquared:.3f}")
print(f"直播电商系数：{model_opt1.params['live_ecom']:.4f}，p值={model_opt1.pvalues['live_ecom']:.4f}（{'显著' if model_opt1.pvalues['live_ecom']<0.05 else '不显著'}）")

# 方案2：面板数据固定效应（更严谨，控制省份个体差异）
data["province_code"] = pd.Categorical(data["province"]).codes  # 省份编码
X_panel = data[["live_ecom", "internet", "province_code"]]
X_panel = sm.add_constant(X_panel)
model_panel = sm.OLS(y, X_panel).fit()
print("\n【方案2：面板数据（控制省份差异）】")
print(f"R² = {model_panel.rsquared:.3f}")
print(f"直播电商系数：{model_panel.params['live_ecom']:.4f}，p值={model_panel.pvalues['live_ecom']:.4f}（{'显著' if model_panel.pvalues['live_ecom']<0.05 else '不显著'}）")

# -------------------------- 6. 可视化优化 --------------------------
# 6.1 分年份散点图
plt.figure(figsize=(8, 5))
for year, color in zip([2021, 2022], ["#1f77b4", "#ff7f0e"]):
    year_data = data[data["year"] == year]
    plt.scatter(year_data["live_ecom"], year_data["gdp_growth"], 
                color=color, label=f"{year}年", alpha=0.8)
plt.xlabel("直播电商交易额（亿元）")
plt.ylabel("GDP增速（%）")
plt.title("2021-2022年直播电商 vs 区域经济增长")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("./output/corr_scatter.png")
plt.show()

# 6.2 2022年空间插值图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 直播电商插值
im1 = ax1.imshow(z_ecom_interp, extent=[100, 125, 20, 40], origin="lower", cmap="YlOrRd")
ax1.scatter(lon, lat, c="black", s=15, label="样本省份")
ax1.set_xlabel("经度（°E）")
ax1.set_ylabel("纬度（°N）")
ax1.set_title("2022年直播电商交易量空间分布")
ax1.legend()
plt.colorbar(im1, ax=ax1, shrink=0.8)

# GDP增速插值
im2 = ax2.imshow(z_gdp_interp, extent=[100, 125, 20, 40], origin="lower", cmap="YlGnBu")
ax2.scatter(lon, lat, c="black", s=15, label="样本省份")
ax2.set_xlabel("经度（°E）")
ax2.set_ylabel("纬度（°N）")
ax2.set_title("2022年GDP增速空间分布")
ax2.legend()
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig("./output/kriging_spatial_dist.png")
plt.show()