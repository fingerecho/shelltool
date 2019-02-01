from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# https://blog.csdn.net/cherdw/article/details/54971771


X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
print("X.shape=%s" % (str(X.shape)))
print("y.shape=%s" % (str(y.shape)))

regr.fit(X, y)

RandomForestRegressor(
    n_estimators=100,  # 太大容易过拟合，太小欠拟合
    bootstrap=True,  # 取样是否放回
    oob_score=True,  # 是否采用袋外样本来评估模型的好坏
    criterion="mse",  # 分类 是 gini ，entropy ， 回归是 mse(均方差)  mae(绝对值差)
    max_depth=64,  # 数据量大时 限制决策树最大深度
    max_features="auto",  # 考虑的最大特征数
    min_samples_split=2,  # 子树划分限制， 少于 这个值 不再细分
    min_samples_leaf=1,  # 叶子节点少于这个值，则和兄弟节点一起剪枝
    max_leaf_nodes=None,  # 通过限制最大叶子节点数，可以防止过拟合
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    min_weight_fraction_leaf=0.0,
    n_jobs=None,
    random_state=0,
    verbose=0,
    warm_start=False,
)

print(regr.feature_importances_)
# [0.18146984 0.81473937 0.00145312 0.00233767]
print(regr.predict([[0, 0, 0, 0]]))
# [-8.32987858]
