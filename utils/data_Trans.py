import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def pk2csv():
    read_path = r'E:\projects\AES\ProTACT\data\cross_prompt_attributes\1\train.pk'
    # 读取 train.pk 文件
    with open(read_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)

    print(train_essays_list[0])
    '''
    {'essay_id': '7532', 'prompt_id': '3', 'score': '1', 'content': '1', 'prompt_adherence': '1', 'language': '1', 'narrativity': '1', 'content_text': 'The features of the setting affected the cyclist in many ways. One way is that the cyclist not sticking to his map. Instead of sticking to his map he had asked a couple of old people. The old people that gave the cyclist directions also gave him a short cut to take. That�s where the second effect came in where the cyclist started to use a shortcut instead of the main road. That�s how the features of the setting affected the cyclist.   '}
    '''
    # # 将 train_essays_list 转换为 DataFrame
    # df = pd.DataFrame(train_essays_list)
    #
    # # 将 DataFrame 写入 CSV 文件
    # df.to_csv('train.csv', index=False)


def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features


def count():
    for i in range(1, 9):
        read_path = fr'E:\projects\AES\ProTACT\data\cross_prompt_attributes\{i}\train.pk'
        # 读取 train.pk 文件
        with open(read_path, 'rb') as train_file:
            train_essays_list = pickle.load(train_file)
        # print(type(train_essays_list),len(train_essays_list))# <class 'list'> 9513
        num1 = len(train_essays_list)

        read_path = fr'E:\projects\AES\ProTACT\data\cross_prompt_attributes\{i}\dev.pk'
        # 读取 train.pk 文件
        with open(read_path, 'rb') as train_file:
            train_essays_list = pickle.load(train_file)
        # print(type(train_essays_list), len(train_essays_list))  # <class 'list'> 1680
        num2 = len(train_essays_list)

        read_path = fr'E:\projects\AES\ProTACT\data\cross_prompt_attributes\{i}\test.pk'
        # 读取 train.pk 文件
        with open(read_path, 'rb') as train_file:
            train_essays_list = pickle.load(train_file)
        # print(type(train_essays_list), len(train_essays_list))  # <class 'list'> 1783
        num3 = len(train_essays_list)
        all = num1 + num2 + num3# =12976
        print(i, 'train/all=', num1 / all, 'dev/all=', num2 / all, 'test/all=', num3 / all, )


'''==========================训练集/==================================='''


def csv2pk():
    '''
    {'essay_id': '7532', 'prompt_id': '3', 'score': '1', 'content': '1', 'prompt_adherence': '1', 'language': '1', 'narrativity': '1', 'content_text': 'The features of the setting affected the cyclist in many ways. One way is that the cyclist not sticking to his map. Instead of sticking to his map he had asked a couple of old people. The old people that gave the cyclist directions also gave him a short cut to take. That�s where the second effect came in where the cyclist started to use a shortcut instead of the main road. That�s how the features of the setting affected the cyclist.   '}
    1.读取xxx.all.csv最后一行，得到CandidateID->essay_id，PromptID->prompt_id, content_text
    2.读取 两种特征分数
    '''
    base_dir = r'E:\projects\keystroke\keystroke_data'
    dirs = ['AC', 'AG', 'AS', 'BB', 'BC', 'BS']
    # Prompt转为prompt_id
    topic2prompt_id = {
        'AC': 1, 'AG': 2, 'AS': 3, 'BB': 4, 'BC': 5, 'BS': 6
    }
    for d in dirs:
        print(d)
        data_list = []
        score1_dic, score2_dic = getScoreDic(d)  # 获取两种特征分数
        read_dir = os.path.join(base_dir, d)
        filenames = os.listdir(read_dir)
        for filename in tqdm(filenames):
            if 'csv' not in filename:  # 不是对应的文件类型
                continue
            else:
                data_dic = {}
                data_df = pd.read_csv(os.path.join(read_dir, filename))
                # 获取essay_id，prompt_id，content_text todo 注意：essay_id=d+CandidateID，是因为CandidateID不唯一，考生会做两个题目
                data_dic['essay_id'] = d + data_df['CandidateID'].tolist()[-1]
                data_dic['prompt_id'] = topic2prompt_id[d]
                data_dic['content_text'] = data_df['TextToDate'].tolist()[-1]
                # 两种特征分数
                content_score = score1_dic.get(data_df['CandidateID'].tolist()[-1], None)  # 内容评分
                conventions_score = score2_dic.get(data_df['CandidateID'].tolist()[-1], None)  # 书面语规范评分
                # todo 数据缺失，补充或者删除
                if content_score:  # content_score不为None值
                    data_dic['content'] = content_score
                else:
                    data_dic['content'] = conventions_score

                if conventions_score:  # content_score不为None值
                    data_dic['conventions'] = conventions_score
                else:
                    data_dic['conventions'] = content_score

            # 把数据添加到data_list
            data_list.append(data_dic)

        '''划分数据集：按照8:1:1'''
        # 确保data_list至少有10个元素以便8:1:1的划分
        assert len(data_list) >= 10, "data_list needs to have at least 10 elements."

        # 随机打乱数据
        np.random.shuffle(data_list)

        # 计算每个部分的大小
        total_len = len(data_list)
        train_len = int(total_len * 0.8)
        dev_len = int(total_len * 0.1)
        test_len = total_len - train_len - dev_len

        # 划分数据集
        train_data = data_list[:train_len]
        dev_data = data_list[train_len:train_len + dev_len]
        test_data = data_list[train_len + dev_len:]

        '''把三个数据集存为pk文件'''
        save_dir = os.path.join(r'D:\zest\keystroke\data', str(topic2prompt_id[d]))

        # 存储为pk文件的函数
        def save_as_pk(data, filename):
            with open(filename, 'wb') as file:
                pickle.dump(data, file)

        # 保存数据
        train_path = os.path.join(save_dir, 'train.pk')
        save_as_pk(train_data, train_path)
        dev_path = os.path.join(save_dir, 'dev.pk')
        save_as_pk(dev_data, dev_path)
        test_path = os.path.join(save_dir, 'test.pk')
        save_as_pk(test_data, test_path)


def getScoreDic(prompt):
    score1 = prompt + 'I'
    score1_df = pd.read_csv(rf'E:\projects\keystroke\keystroke_data\score\{score1}.csv',
                            usecols=['CANDIDATE_ID', score1])
    # 将CANDIDATE_ID列设置为索引，score1列作为对应的值
    score1_dic = score1_df.set_index('CANDIDATE_ID')[score1].to_dict()

    score2 = prompt + 'II'
    score2_df = pd.read_csv(rf'E:\projects\keystroke\keystroke_data\score\{score2}.csv',
                            usecols=['CANDIDATE_ID', score2])
    # 将CANDIDATE_ID列设置为索引，score2列作为对应的值
    score2_dic = score2_df.set_index('CANDIDATE_ID')[score2].to_dict()
    # print(score1_dic, score2_dic)
    return score1_dic, score2_dic


'''=================================手工特征============================================'''
'''
E:\projects\AES\ProTACT\data\LDA\hand_crafted_final_1.csv
item_id,prompt_id,highest_topic,mean_word,word_var,mean_sent,sent_var,ess_char_len,word_count,prep_comma,unique_word,clause_per_s,mean_clause_l,max_clause_in_s,spelling_err,sent_ave_depth,ave_leaf_depth,automated_readability,linsear_write,stop_prop,positive_sentence_prop,negative_sentence_prop,neutral_sentence_prop,overall_positivity_score,overall_negativity_score,",",.,VB,JJR,WP,PRP$,VBN,VBG,IN,CC,JJS,PRP,MD,WRB,RB,VBD,RBR,VBZ,NNP,POS,WDT,DT,CD,NN,TO,JJ,VBP,RP,NNS,score
'''


def make_hand_features():
    '''
    item_id=
    prompt_id=
    特征构造：
    sequence_count：操作序列总数
    writing_duration：写作时长
    jump_count：跳转操作的次数
    delete_count：删除操作的次数
    add_count：insert操作的次数
    -EditingEvent：LastWordEdited，retyping，BackspacedMultiWordChunk
    tasa_frequence_min，tasa_frequence_mean，tasa_frequence_max：使用不为0的求最小值，均值，最大值
    KeyboardingState：统计各个类型的值的数量，['InWord' 'BetweenWord' 'BackSpace' 'Edit' 'BetweenSentence' 'BetweenParagraph']
    essay_words_count：文章单词总数
    score1：论文评分
    score2：论文评分
    '''
    base_dir = r'E:\projects\keystroke\keystroke_data'
    dirs = ['AC', 'AG', 'AS', 'BB', 'BC', 'BS']
    # Prompt转为prompt_id
    topic2prompt_id = {
        'AC': 1, 'AG': 2, 'AS': 3, 'BB': 4, 'BC': 5, 'BS': 6
    }
    for d in dirs:
        print(d)
        # 数据列表
        item_id, prompt_id, sequence_count, writing_duration, jump_count, delete_count, add_count, tasa_frequence_min, \
        tasa_frequence_mean, tasa_frequence_max, keyboardingState_InWord, keyboardingState_BetweenWord, keyboardingState_BackSpace, keyboardingState_Edit, \
        keyboardingState_BetweenSentence, keyboardingState_BetweenParagraph, essay_words_count, score = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        score1_dic, score2_dic = getScoreDic(d)  # 获取两种特征分数
        read_dir = os.path.join(base_dir, d)
        filenames = os.listdir(read_dir)
        for filename in tqdm(filenames):
            if 'csv' not in filename:  # 不是对应的文件类型
                continue
            else:
                data = pd.read_csv(os.path.join(read_dir, filename))  # 读取数据
                # todo 注意：essay_id=d+CandidateID，是因为CandidateID不唯一，考生会做两个题目
                item_id.append(d + data['CandidateID'][0])
                prompt_id.append(topic2prompt_id[d])

                sequence_count.append(len(data['sequenceIndex']))  # 操作序列总数
                writing_duration.append(float(data['TimeStamp'].tolist()[-1]) - float(data['TimeStamp'][0]))  # 写作时长
                jump_count.append(data['IsJump'].sum())  # 跳转操作的次数
                delete_count.append(len(data[data['Operation'] == 'Delete']))  # Delete操作的次数
                add_count.append(len(data[data['Operation'] == 'Insert']))  # insert操作的次数
                tasa_frequence = data[data['TASA_FREQUENCY'] != 0]
                # 使用tasa_frequence(不为0)的求最小值，均值，最大值
                tasa_frequence_min.append(tasa_frequence['TASA_FREQUENCY'].min())
                tasa_frequence_mean.append(tasa_frequence['TASA_FREQUENCY'].mean())
                tasa_frequence_max.append(tasa_frequence['TASA_FREQUENCY'].max())
                # 使用KeyboardingState统计 ['InWord' 'BetweenWord' 'BackSpace' 'Edit' 'BetweenSentence' 'BetweenParagraph']数量
                keyboardingState_InWord.append(len(data[data['KeyboardingState'] == 'InWord']))
                keyboardingState_BetweenWord.append(len(data[data['KeyboardingState'] == 'BetweenWord']))
                keyboardingState_BackSpace.append(len(data[data['KeyboardingState'] == 'BackSpace']))
                keyboardingState_Edit.append(len(data[data['KeyboardingState'] == 'Edit']))
                keyboardingState_BetweenSentence.append(len(data[data['KeyboardingState'] == 'BetweenSentence']))
                keyboardingState_BetweenParagraph.append(len(data[data['KeyboardingState'] == 'BetweenParagraph']))
                # 文章单词总数
                essay_words_count.append(len(data['TextToDate'].tolist()[-1].strip().split(' ')))
                # 论文评分 todo 这个score代表的是总体评分，在这里不重要，使用时会删除。
                # content_score,conventions_score 两种特征分数
                content_score = score1_dic.get(data['CandidateID'].tolist()[-1], None)  # 内容评分
                conventions_score = score2_dic.get(data['CandidateID'].tolist()[-1], None)  # 书面语规范评分
                if content_score:  # content_score不为None值
                    score.append(content_score)
                else:
                    score.append(conventions_score)

        # 数据字典
        data_dic = {
            'item_id': item_id,
            'prompt_id': prompt_id,
            'sequence_count': sequence_count,
            'writing_duration': writing_duration,
            'jump_count': jump_count,
            'delete_count': delete_count,
            'add_count': add_count,
            'tasa_frequence_min': tasa_frequence_min,
            'tasa_frequence_mean': tasa_frequence_mean,
            'tasa_frequence_max': tasa_frequence_max,
            'keyboardingState_InWord': keyboardingState_InWord,
            'keyboardingState_BetweenWord': keyboardingState_BetweenWord,
            'keyboardingState_BackSpace': keyboardingState_BackSpace,
            'keyboardingState_Edit': keyboardingState_Edit,
            'keyboardingState_BetweenSentence': keyboardingState_BetweenSentence,
            'keyboardingState_BetweenParagraph': keyboardingState_BetweenParagraph,
            'essay_words_count': essay_words_count,
            'score': score,
        }
        data_df = pd.DataFrame(data_dic)
        save_path=os.path.join(r'D:\zest\keystroke_processed_data\keystroke_hand_crafted_feature',f'hand_crafted_final_{topic2prompt_id[d]}.csv')
        data_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    # 数据集构造
    # csv2pk()
    # 手工特征构造
    # make_hand_features()
    pass
