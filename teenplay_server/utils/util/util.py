import os
from pathlib import Path

import joblib

from ai.models import ReplyAi


def check_the_comments(reply_content):
    result = 'success'
    # 모델소환
    model_file_path = os.path.join(Path(__file__).resolve().parent.parent.parent.parent, 'ai/ai/reply_default_model.pkl')
    model = joblib.load(model_file_path)
    X_train = [reply_content]
    prediction = model.predict(X_train)

    if prediction[0] == 1:
        # 추가 fit
        transformed_X_train = model.named_steps['count_vectorizer'].transform(X_train)
        model.named_steps['multinomial_NB'].partial_fit(transformed_X_train, prediction)
        joblib.dump(model, model_file_path)

        # insert
        ReplyAi.objects.create(comment=X_train[0], target=prediction[0])
        result = 'profanity'

    return result