from kiwipiepy import Kiwi

from transformers import ElectraModel, ElectraTokenizer
from transformers import ElectraForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSemanticSegmentation, TrainingArguments, Trainer

import torch
from torch.nn.functional import softmax

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import pandas as pd
import re

import warnings
warnings.filterwarnings('ignore')

class escape:
    def __init__(self):
        category_names = [
            'guide', 'light', 'interior', 'story', 'probability', 'creativity', 'production', 'activity', 'scale', 'fear', 'device', 'fun', 'service'
            ]
        # 객체의 데이터 값은 survey 데이터에서 읽어온다.
        self.survey = self.read_data("./data/survey.csv")
        self.review = pd.read_csv("./data/reviews.csv", index_col=0)

        # model과 tokenizer 정의
        self.model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=3, problem_type="multi_label_classification")
        # 추가된 단어로 만들어진 토크나이저 사용
        self.tokenizer = self.add_token("./add_data.txt")

    def add_token(self, path):
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        # 토큰에 추가할 단어 -> '방탈출'이라는 도메인 지시기에 근거한 용어, 분리되어서는 안 되기 때문에 별도로 추가 작업 진행
        with open(path, "r") as add:
            add_data = add.read()
        addword = add_data.split('\n')
        # token에 새로운 단어 추가 
        tokenizer.add_tokens(addword)
        # token에 단어 추가후 기존 모델의 임베딩 레이어에 추가한 단어에 대한 임베딩 벡터가 없을 수 있기 때문
        # 아래 코드를 통해서 토큰의 개수가 변했음을 모델에 알리고 모델의 임베딩 레이어를 조정하여 새로운 토큰을 수용할 수 있게 함
        self.model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    def read_data(self, path):
        survey = pd.read_csv(path, index_col=0)
        survey.fillna(0, inplace=True)
        survey = survey.reset_index()
        return survey

    # 최소한의 전처리
    def cleaned_content(self, text):
        d = re.sub('\n', '. ', text) # 줄바꿈 > .
        d = re.sub('[^가-힣0-9a-zA-Z ]{2,}', ".", d) # 특수문자 두개 이상인거 .으로 변경
        return d
    
    # 키위 전처리
    def kiwi_clean(self, text):
        kiwi = Kiwi()
        get_kiwi_pos = ['NNG', 'NP', 'NNP', 'MM', 'VV', 'VV-I', 'VV-R', 'VA', 'VA-I', 'VA-R', 'VCP', 'VCN', 'MAG', 'MAJ', 'XR']
        kiwi_lem = []
        for word in kiwi.tokenize(text):
            if word.tag in get_kiwi_pos:
                kiwi_lem.append(word.lemma)
        return ' '.join(kiwi_lem)

    # survey data에서 특정 target data 가져오기
    def clean_survey(self, target):    
        # survey data 선언
        survey_data = self.survey 
        # target이 되는 data 골라오기
        target_data = survey_data[['content_id', target]]
        # content_id 중에서 댓글이 같이 추가된 데이터 정리하기
        target_data['content_id'] = target_data['content_id'].apply(lambda x: x.split(',')[0] if len(str(x)) > 6 else x)
        # content_id 가 0인 데이터 제외하기 -> 추후 merge를 위해 content_id 모두 int type으로 변경
        target_data = target_data[target_data['content_id'] != 0].reset_index().drop(columns=['index'])
        target_data['content_id'] = target_data['content_id'].astype(int)
        return target_data

    def final_data(self, target):
        # 전체 review data 불러오기
        review_all = self.review
        # target survey data 선언하기
        target_survey = self.clean_survey(target)
        print(target_survey)
        # 전체 review data 중에서 survey data에 있는 댓글만 가져오기
        survey_content = review_all.loc[review_all['id'].isin(target_survey['content_id'])][['id', 'content']]
        # merge 를 위해서 id 컬럼명 통일하기
        survey_content.columns=['content_id', 'content']
        # 통일된 content_id 를 기반으로 데이터 merge
        fin_df = pd.merge(target_survey, survey_content)
        print(fin_df)
        # 추후 원활한 계산을 위해서 숫자 부분은 모두 int 로 바꿔줌
        fin_df[target] = fin_df[target].astype(int)
        fin_df['content'] = fin_df['content'].apply(self.cleaned_content)
        fin_df['content'] = fin_df['content'].apply(self.kiwi_clean)
        # 최종 데이터
        final_df = fin_df[['content', target]]
        # 최종 데이터 중복 제거
        final_df = final_df.drop_duplicates()
        print(final_df)
        return final_df


es = escape()
es.final_data('story')