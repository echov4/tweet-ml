from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class TweetForm(FlaskForm):
    tweet = StringField('Tweet', validators=[Length(min=1, max=280)])
    submit = SubmitField("Classify Tweet")