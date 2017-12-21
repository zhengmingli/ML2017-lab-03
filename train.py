import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
from scipy import misc  #对图像进行缩放
from feature import NPDFeature
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

sampleSize = 400
maxIteration = 10 #分类器个数

img =[]
img_label = []
img_features = []
img_label_train, img_label_validation = [],[]
img_features_train, img_features_validation = [],[]

def load_img_data():
    # img_face
    for i in range(0,int(sampleSize/2)):
        image = mpimg.imread("./datasets/original/face/face_"+"{:0>3d}".format(i)+".jpg")
        image_gray = rgb2gray(image)
        image_gray_scaled = misc.imresize(image_gray,(24,24))
        img.append(image_gray_scaled) 
        img_label.append(1)

        image = mpimg.imread("./datasets/original/nonface/nonface_"+"{:0>3d}".format(i)+".jpg")
        image_gray = rgb2gray(image)
        image_gray_scaled = misc.imresize(image_gray,(24,24))
        img.append(image_gray_scaled)
        img_label.append(-1)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def extra_img_features():
    for i in range(0,len(img)):
        f = NPDFeature(img[i])
        features = f.extract()
        img_features.append(features)
        
def get_accuracy(pred, y):
    return sum(pred == y) / float(len(y))

def get_error_rate(pred,y):
    return sum(pred != y) / float(len(y))

if __name__ == "__main__":

    #加载数据
    load_img_data()

    # 将预处理后的特征数据保存到缓存中
    # with open('data', "wb") as f:
    #    extra_img_features()
    #    pickle.dump(img_features, f)

    # 使用load()函数读取特征数据
    with open('data', "rb") as f:
        img_features = pickle.load(f)

    img_label_train = img_label[0:int(sampleSize*0.7)]
    img_label_validation = img_label[int(sampleSize*0.7):]
    img_features_train = img_features[0:int(sampleSize*0.7)]
    img_features_validation = img_features[int(sampleSize*0.7):]

    # 初始化权重
    weights = np.ones(len(img_features_train)) / len(img_features_train)
    hypothesis = []  # 记录每个分类器的hypothesis
    alpha_m = []     # 记录每个分类器的alpha
    accuracy = []    # 记录累积到第i个分类器时的精确率
    clf_tree = DecisionTreeClassifier(max_depth = 3 , random_state = 1)

    prediction = np.zeros(len(img_features_train),dtype=np.int32)

    
    # 训练
    for i in range(0, maxIteration):
        print("Iteration",i)
        clf_tree.fit(img_features_train, img_label_train, sample_weight=weights)
        hypothesis.append (clf_tree.predict(img_features_train) )

        miss = [int(x) for x in ( hypothesis[i] != img_label_train )]
        miss2 = [x if x==1 else -1 for x in miss]
        err_m = np.dot(weights,miss)
        if(err_m > 0.5):
            break
        print('error: ',err_m)
        alpha_m.append( 0.5 * np.log( (1 - err_m) / float(err_m)) )
        weights = np.multiply(weights, np.exp([float(x) * alpha_m[i] for x in miss2]))
        weights_sum = weights.sum()
        weights = weights / weights_sum
        prediction = prediction + alpha_m[i] * hypothesis[i]
        print('train_accuracy: ',get_accuracy(np.sign(prediction),img_label_train))


        validation_pred = clf_tree.predict(img_features_validation)
        validation_prediction = alpha_m[i] * validation_pred
        target_names = ['class 0', 'class 1']
        acc = get_accuracy(np.sign(validation_prediction), img_label_validation)
        accuracy.append(acc)
        print(classification_report(img_label_validation, np.sign(validation_prediction), target_names=target_names))
        with open('./report.txt','a') as report:
            report.write(classification_report(img_label_validation, np.sign(validation_prediction), target_names=target_names))

    # show
    t = np.arange(0, maxIteration, 1)
    plt.plot(t, accuracy, color="red", linewidth=2.5, linestyle="-", label="accuracy")
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()


    


