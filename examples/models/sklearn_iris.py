from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Input shape:', X_test.shape)

logistic = LogisticRegression(solver='liblinear', multi_class='auto')
logistic.fit(X_train, y_train)

logistic_output = logistic.predict(X_test)
print('Logistic output shape:', logistic_output.shape)

linear = LinearRegression()
linear.fit(X_train, y_train)

linear_output = linear.predict(X_test)
print('Linear output shape:', linear_output.shape)

initial_type = [('float_input', FloatTensorType([1, 4]))]

logistic_onnx = convert_sklearn(logistic, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(logistic_onnx.SerializeToString())

linear_onnx = convert_sklearn(linear, initial_types=initial_type)
with open("linear_iris.onnx", "wb") as f:
    f.write(linear_onnx.SerializeToString())

