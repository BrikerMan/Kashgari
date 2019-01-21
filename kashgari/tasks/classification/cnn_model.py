# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_model.py
@time: 2019-01-21 17:49

"""
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.tasks.classification.base_model import ClassificationModel


class CNNModel(ClassificationModel):

    def build_model(self):
        current, input_layers = self.prepare_embedding_layer()
        conv1d = Conv1D(128, 5, activation='relu')(current)
        max_pool = GlobalMaxPooling1D()(conv1d)
        dense_1 = Dense(64, activation='relu')(max_pool)
        dense_2 = Dense(len(self.tokenizer.label2idx), activation='sigmoid')(dense_1)

        model = Model(input_layers, dense_2)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger

    init_logger()

    x_data = ['奇怪，端午节了，怎么没看到超市里有月饼卖啊？@ 王子26', '"哥，你闷骚！年轻不知精珍贵',
              '中美之间的汇率大战之中最阴险的就是美国在各国运作，给中国压力希望人民币升值，而我们不能升值的结果就是让步。其实，我们应该学习德国治理马克的经验，马克兑美元由4马克兑1美升到1.5马克对1美元，他们的经济依然平稳过渡，这就是水平，而水平决定一切。',
              '可惜我没听到呀，遗憾我就知道每次看完包子脑子都是一片空白，不过它会慢慢浮现的....好久没有这么近距离的听包子唱歌了，包子还是这么帅，歌声还是这么天籁~ 又看见了一些久未见面的亲们，和一些新的亲们，很开心。星星们今天赞一个，等到这么晚都还在坚持，这就是她们!明天就要去送包子机了，很舍不得....',
              '无【围观三亚人类史上最庞大、最浪费惊人的拆违大会战】上千万平方米的小区楼盘盖起来，再拆掉，再盖起来，三亚的gdp 估计会全国之一。有网友称，三亚此役将记入中国历史，和当年百团大战齐名。',
              '有个国外的视频，就是在这种卫生间里恶搞的，又尴尬又开心“万一门锁故障不能断电，岂非让人看光光！”还真不好说。',
              '回复这样的校长和老师教出来的学生肯定都信佛的无奇不有呀！大新闻，第一次听说大新闻，第一次听说武汉市新洲一所初中学校日前组织30余名九年级老师参加敬香祈福，希望九年级的孩子们考出好成绩。一位副校长表示，每年都会组织毕业年级班主任敬香，这已是常规工作。---“考前抱佛脚”于事无补、不要放大对考试的焦虑、老师的压力需要一个出口、关键要扭转应试倾向。',
              '回复我擦。。好吧。。我宝宝我才不让给你呢。你想的美。。就是这个博主啊……我大姐到底是谁啊。。我还没搞清楚。。文艺神马的。。。人身處 在幽靜 單 純 的城市。心也會 慢慢安靜 下來 。',
              '这世道！这国家迟早…维权网唐奇虎报道拆迁公司的流氓有恃无恐一直骚扰纠缠到第二天凌晨2：30左右，还在不停地谩骂，最后终于大出大手，四十余人一起下手，对陈刚拳脚相加，并用条凳砸陈刚的脑颅。陈刚被打得血流满面，白色衬衫全被鲜血染红，陈刚被逼无奈，拿起菜刀进陈刚正在南通市附属医院接受抢救治疗。',
              '每个人都有许多小秘密，我把它藏进梦里……']
    y_data = ['低落', '喜悦', '喜悦', '喜悦', '愤怒', '喜悦', '喜悦', '喜悦', '愤怒', '愤怒']

    classifier = CNNModel()
    classifier.fit(x_data, y_data, batch_size=2)