import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 생성
n_features = 40
tr_cf_num = 6000
tr_nc_num = 11000
te_cf_num = 3000
te_nc_num = 7000
tr_fracture_data = np.random.normal(loc=0.5, scale=1, size=(tr_cf_num, n_features))  # Compression Fracture group
tr_normal_control_data = np.random.normal(loc=-0.5, scale=1, size=(tr_nc_num, n_features))  # Normal Control group
te_fracture_data = np.random.normal(loc=0.2, scale=1.3, size=(te_cf_num, n_features))  # Compression Fracture group
te_normal_control_data = np.random.normal(loc=-0.2, scale=1.3, size=(te_nc_num, n_features))  # Normal Control group

# 각 그룹 데이터를 결합
data = np.concatenate([tr_fracture_data, tr_normal_control_data, te_fracture_data, te_normal_control_data], axis=0)

# 샘플 그룹 라벨 생성 (순서대로 설정)
groups = ['Compression Fracture'] * tr_cf_num + ['Normal Control'] * tr_nc_num + ['Compression Fracture'] * te_cf_num + ['Normal Control'] * te_nc_num

# DataFrame으로 변환
feature_names = [f"feature_{i+1}" for i in range(n_features)]
df = pd.DataFrame(data, columns=feature_names)

# 그룹 정보를 색상으로 표시
group_colors = pd.Series(groups).map({'Compression Fracture': 'red', 'Normal Control': 'blue'})

# 히트맵 생성 (X축 = 순서대로 tr_fracture, tr_normal, te_fracture, te_normal)
# plt.figure(figsize=(12, 8))
sns.clustermap(df.T, method='ward', metric='euclidean', cmap="coolwarm", row_cluster=True, col_cluster=False, 
               col_colors=group_colors, figsize=(12,8))
# plt.title("Hierarchical Clustering Heatmap of Radiomics Features (Ordered Groups on X-axis, Features on Y-axis)")
plt.show()
