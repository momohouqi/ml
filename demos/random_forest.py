from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

batch_x = x_train[:50000].reshape(50000, -1)
batch_y = y_train[:50000]
test_x = x_test[:10000].reshape(10000, -1)
test_y = y_test[:10000]
print('start random forest')
for i in range(10, 200, 10):
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(batch_x, batch_y)

    y_predict = rf.predict(test_x)
    acc = accuracy_score(test_y, y_predict)
    print("n_estimators = {}, acc:{}".format(i, acc))