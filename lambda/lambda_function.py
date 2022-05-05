import boto3
import json
import email
import os

ENDPOINT = os.environ['SageMakerEndPoint']
print(ENDPOINT)

############### UTILS ###############
import string
import sys
import numpy as np

from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):

    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

############### END ###############

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    s3 = boto3.client('s3')
    messageRaw = s3.get_object(Bucket=bucket, Key=key)

    emailObj = email.message_from_bytes(messageRaw['Body'].read())
    femail = emailObj.get('From')
    body = emailObj.get_payload()[0].get_payload()
    date_var = emailObj.get('date')
    

    print(femail)
    print(body)

    sgEndpoint = ENDPOINT
    print(ENDPOINT)
    sruntime = boto3.client('runtime.sagemaker')

    #model input formatiing
    vocabulary_length = 9013
    input_mail = [body.strip()]
    one_hot_encode_data = one_hot_encode(input_mail, vocabulary_length)

    preProcessedInputMail = vectorize_sequences(one_hot_encode_data, vocabulary_length)
    jdata = json.dumps(preProcessedInputMail.tolist())

    #check for spam
    modelResponse = sruntime.invoke_endpoint(EndpointName=sgEndpoint, ContentType='application/json', Body=jdata)
    result = json.loads(modelResponse["Body"].read())

    if result['predicted_label'][0][0] == 0:
        label = 'Ok'
    else:
        label = 'Spam'
    
    prediction = round(result['predicted_probability'][0][0], 4)
    prediction = prediction*100

    print("Spam: ",label)
    print("Prediction: ", prediction)

    remail = emailObj.get('To')
    message = "We received your email sent at " + str(remail) + " with the subject " + str(emailObj.get('Subject')) + " on the given date: " + date_var +"."+"\nHere is a 240 character sample of the email body:\n\n" + body[:240] + "\nThe email was categorized as " + str(label) + " with a " + str(prediction) + "% confidence."
    
    email_client = boto3.client('ses')
    reply_email = email_client.send_email(
        Destination={'ToAddresses': [femail]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(remail),
    )

    return {}