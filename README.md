AI Challenger 2018 Sentiment Analysis Baseline with fastText
=========================================
功能描述
---
本项目主要基于AI Challenger官方[baseline](https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline)修改了一个基于fastText的baseline，方便参赛者快速上手比赛，主要功能涵盖完成比赛的全流程，如数据读取、分词、特征提取、模型定义以及封装、
模型训练、模型验证、模型存储以及模型预测等。baseline仅是一个简单的参考，希望参赛者能够充分发挥自己的想象，构建在该任务上更加强大的模型。

开发环境
---
* 主要依赖工具包以及版本，详情见requirements.txt

项目结构
---
* src/config.py 项目配置信息模块，主要包括文件读取或存储路径信息
* src/util.py 数据处理模块，主要包括数据的读取以及处理等功能
* src/main_train.py 模型训练模块，模型训练流程包括 数据读取、分词、特征提取、模型训练、模型验证、模型存储等步骤
* src/main_predict.py 模型预测模块，模型预测流程包括 数据和模型的读取、分词、模型预测、预测结果存储等步骤


使用方法
---
* 准备 virtualenv -p python3 venv & source venv/bin/activate & pip install -r requirement.txt
* 配置 在config.py中配置好文件存储路径
* 训练 运行 python main_train.py -mn your_model_name 训练模型并保存，同时通过日志可以得到验证集的F1_score指标
* 预测 运行 python main_predict.py -mn your_model_name 通过加载上一步的模型，在测试集上做预测
* 更多详情请参考我的博客文章：http://www.52nlp.cn/?p=10537

以下是我在家里这台深度学习机器上的测试，不过并没有用到GPU，主要用的是CPU和内存：CPU是Intel/英特尔 Xeon E5-1620V4 CPU 4核心8线程，内存是48G。因为 skift 只支持python3, 所以是在Ubuntu16.04, python3.5的环境下测试的，其他环境是否能顺利测试通过不清楚。
git clone https://github.com/panyang/fastText-for-AI-Challenger-Sentiment-Analysis.git
cd fastText-for-AI-Challenger-Sentiment-Analysis/
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirement.txt

注意准备工作做完后，需要在config里设置相关文件的路径，默认配置大概是这样的：
import os

data_path = os.path.abspath('..') + "/data"
model_path = data_path + "/model/"
train_data_path = data_path + "/train/train.csv"
validate_data_path = data_path + "/valid/valid.csv"
test_data_path = data_path + "/test/testa.csv"
test_data_predict_output_path = data_path + "/predict/testa_predict.csv"

注意运行代码前需要将相关输入输出路径建好，并将相关数据路径指定好，可以直接拷贝AI Challenger官方提供的数据文件，也可以用软连接指向。如果不做任何设定，可以直接用默认配置参数训练fasttext多模型，直接运行“python main_train.py” 即可。这样大概跑了不到10分钟，内存峰值占用不到8G，在验证集上得到一个f1均值约为0.5451的fasttext多分类模型(20个），模型存储位置在 model_path 下：fasttext_model.pkl，大概1.8G，在验证集上详细的F1值大致如下：
location_traffic_convenience:0.5175700387941342
location_distance_from_business_district:0.427891674259875
location_easy_to_find:0.570805555583767
service_wait_time:0.5052181634999748
service_waiters_attitude:0.6766570408968818
service_parking_convenience:0.5814636947460745
service_serving_speed:0.5701241141533907
price_level:0.6161258412096242
price_cost_effective:0.5679586399625348
price_discount:0.5763345656700684
environment_decoration:0.5554146717297597
environment_noise:0.563452055291662
environment_space:0.5288336794721515
environment_cleaness:0.5511776910510577
dish_portion:0.5527095496220675
dish_taste:0.6114994440656155
dish_look:0.43750894239614163
dish_recommendation:0.41756941548558957
others_overall_experience:0.5322283082904627
others_willing_to_consume_again:0.5404900044311536


2018-10-02 14:32:18,927 [INFO]  (MainThread) f1_score: 0.5450516545305993

这是默认参数跑出来的结果，另外代码里增加了一些fastText可用的参数选项，例如learning_rate，word_ngrams, min_count参数，也可以基于这些参数进行调参：
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        default='fasttext_model.pkl',
                        help='the name of model')
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='?',
                        default=1.0)
    parser.add_argument('-ep', '--epoch', type=int, nargs='?',
                        default=10)
    parser.add_argument('-wn', '--word_ngrams', type=int, nargs='?',
                        default=1)
    parser.add_argument('-mc', '--min_count', type=int, nargs='?',
                        default=1)

    args = parser.parse_args()
    model_name = args.model_name
    learning_rate = args.learning_rate
    epoch = args.epoch
    word_ngrams = args.word_ngrams
    min_count = args.min_count
    
作为一个例子，这里将word_ngrams设置为2再跑一次：

python main_train.py -mn fasttext_wn2_model.pkl -wn 2

这次大约跑了15分钟，内存峰值最大到37G，存储的模型大约在17G，验证集F1值结果如下：
location_traffic_convenience:0.5482785384602362
location_distance_from_business_district:0.4310319720574882
location_easy_to_find:0.6140713866422334
service_wait_time:0.5247890022873511
service_waiters_attitude:0.6881098513108542
service_parking_convenience:0.5828935095474249
service_serving_speed:0.6168828054420539
price_level:0.6615100420842464
price_cost_effective:0.5954569043369508
price_discount:0.5744529736585073
environment_decoration:0.5743996877298929
environment_noise:0.6186211367923496
environment_space:0.5981761036053918
environment_cleaness:0.6002515744280692
dish_portion:0.5733503000134572
dish_taste:0.6075507493398153
dish_look:0.4424685719881029
dish_recommendation:0.5936671419596734
others_overall_experience:0.5325664419580063
others_willing_to_consume_again:0.5875683298630815

2018-10-02 14:53:00,701 [INFO]  (MainThread) f1_score: 0.5783048511752592

这个结果看起来还不错，我们可以基于这个fasttext多分类模型进行测试集的预测：

python main_predict.py -mn fasttext_wn2_model.pkl

大约运行不到3分钟，预测结果就出炉了，可以在 test_data_predict_output_path 找到这个预测输出文件: testa_predict.csv ，然后就可以去官网提交了，在线提交的结果和验证集F1值大致相差0.01~0.02。这里还可以做一些事情来优化结果，譬如去停用词，不过我试了去停用词和去一些标点符号，结果还有一些降低；调参，learning_rate的影响是比较直接的，min_count设置为2貌似也有一些负向影响，有兴趣的同学可以多试试，寻找一个最优组合。
