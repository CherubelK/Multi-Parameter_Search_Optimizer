import torch
import pandas as pd


def Initialize_tokenizer():
    from transformers import DistilBertTokenizerFast
    # Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return tokenizer


def Token_Classification_Initialization():
    from transformers import DistilBertForTokenClassification
    Token_Classification_Labels = {
                                  'O': 0,
                                  'B-Product': 1,
                                  'I-Product': 2,
                                  'B-Material': 3,
                                  'I-Material': 4,
                                  'B-Country': 5,
                                  'I-Country': 6,
                                  'B-Application': 7,
                                  'I-Application': 8,
                                  'B-Recycle': 9,
                                  'I-Recycle': 10
                                  }
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Token_Classification_model = DistilBertForTokenClassification.from_pretrained(
                                    'distilbert-base-uncased',
                                    num_labels=len(Token_Classification_Labels)
                                                                                ).to(device)
    return Token_Classification_Labels, Token_Classification_model


def Token_Classification_Results(List_of_Sentences: list,
                                 Token_Classification_Labels: dict,
                                 Token_Classification_model,
                                 tokenizer,
                                 Token_Class_File: str
                                 ) -> dict:
    # Puts model into eval mode
    Token_Classification_model.eval()

    # Loads the saved file of Pre-trained model
    Token_Classification_model.load_state_dict(torch.load(Token_Class_File))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Inverse mapping from label indices to labels
    inv_label_map = {v: k for k, v in Token_Classification_Labels.items()}

    # Creating Final Dictionary for results
    final_dict = {
        'Product Type': [],
        'Material Type': [],
        'Building applications': [],
        'Countries': [],
        'Recycled Content': []
    }

    # Cycling through query
    for sentence in List_of_Sentences:
        # Encode the sentence
        inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            padding='longest',
            return_token_type_ids=True
        )

        # Create torch tensors and move them to the device
        input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)

        # Run the sentence through the model
        with torch.no_grad():
            outputs = Token_Classification_model(input_ids, attention_mask=attention_mask)

        # Get the token-level class probabilities
        logits = outputs[0]

        # Compute the predicted labels
        predictions = torch.argmax(logits, dim=-1)

        # Remove padding and special tokens
        input_ids = input_ids[0].tolist()
        predictions = predictions[0].tolist()

        real_predictions = [pred for id, pred in zip(input_ids, predictions) if id != 0 and id != 101 and id != 102]

        # Map predicted label indices back to label strings
        predicted_labels = [inv_label_map[label] for label in real_predictions]

        # Combine tokens and their predicted labels into dict
        results = dict(zip(sentence.split(' '), predicted_labels))

        # Put results into final_dict, key is word and value is what it is labeled as
        for word, label in results.items():
            print(word, label)
            if word in ['for', 'and', 'from', 'in']:
                continue
            if label != 'O':
                # Adds word to proper dictionary if belongs to that type

                # Checks if there is a B-word before the I-word so it will add to it

                # It will make a new word if there is no B-word

                if label == 'B-Product':
                    final_dict['Product Type'].append(word)

                elif label == 'I-Product':

                    if final_dict['Product Type']:
                        final_dict['Product Type'][-1] += ' ' + word
                    else:
                        final_dict['Product Type'].append(word)

                elif label == 'B-Material':
                    final_dict['Material Type'].append(word)

                elif label == 'I-Material':

                    if final_dict['Material Type']:
                        final_dict['Material Type'][-1] += ' ' + word
                    else:
                        final_dict['Material Type'].append(word)

                elif label == 'B-Country':
                    final_dict['Countries'].append(word)

                elif label == 'I-Country':

                    if final_dict['Countries']:
                        final_dict['Countries'][-1] += ' ' + word
                    else:
                        final_dict['Countries'].append(word)

                elif label == 'B-Application':
                    final_dict['Building applications'].append(word)

                elif label == 'I-Application':

                    if final_dict['Building applications']:
                        final_dict['Building applications'][-1] += ' ' + word
                    else:
                        final_dict['Building applications'].append(word)

                elif label == 'B-Recycle':
                    final_dict['Recycled Content'].append(word)

                elif label == 'I-Recycle':

                    if final_dict['Recycled Content']:
                        final_dict['Recycled Content'][-1] += ' ' + word
                    else:
                        final_dict['Recycled Content'].append(word)
        print('\n')

    return final_dict


def label_encoder_initialization(sheet_name: str):
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_excel('..//datasets_and_models/label_encoder.xlsx', sheet_name=sheet_name)
    labels = df['labels'].tolist()
    le = LabelEncoder()
    le.fit_transform(labels)
    return le



def Sequence_Classification_Model_Initialization(le):
    from transformers import DistilBertForSequenceClassification
    Sequence_Classification_Model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))
    return Sequence_Classification_Model, le



def NER_initialization():
    tokenizer = Initialize_tokenizer()
    Token_Classification_Labels, Token_Classification_model = Token_Classification_Initialization()
    return tokenizer, Token_Classification_Labels, Token_Classification_model



def Category_initialization(sheet_name: str):
    Seq_model, le = Sequence_Classification_Model_Initialization(label_encoder_initialization(sheet_name=sheet_name))
    return Seq_model, le



def Category_Matching_Prediction(input_words, Sequence_Classification_Model, tokenizer, le, Sequence_Class_File):
    Sequence_Classification_Model.eval()
    Sequence_Classification_Model.load_state_dict(torch.load(Sequence_Class_File, map_location=torch.device('cpu')))


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    final_output = []

    for word in input_words:
        word = word.lower()

        input_text_encoded = tokenizer(word, truncation=True, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            logits = Sequence_Classification_Model(**input_text_encoded).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_5_probs, top_5_labels = torch.topk(probs, 5)

        top_5_labels = le.inverse_transform(top_5_labels.cpu().detach().numpy()[0])
        top_5_probs = top_5_probs.cpu().detach().numpy()[0]

        predicted_product_types = list(top_5_labels)
        predicted_probabilities = list(top_5_probs)

        predicted_probabilities = [p / sum(predicted_probabilities) for p in predicted_probabilities]

        for pt, prob in zip(predicted_product_types, predicted_probabilities):
            final_output.append((pt, round(prob * 100, 2)))
    return final_output


def find_recycled_content(input_string):
    import re
    # Load an excel sheet containing recycled content categories and their respective keywords into a DataFrame
    df = pd.read_excel('../datasets_and_models/categories_w_keywords.xlsx', sheet_name='recycled_content')

    # Compute the length of the 'keywords' for each category
    df['keywords_len'] = df['keywords'].apply(len)
    # Sort the dataframe based on the length of 'keywords' in descending order
    # This ensures that longer and more specific keywords are matched first
    df = df.sort_values('keywords_len', ascending=False)

    # Remove the auxiliary 'keywords_len' column
    df = df.drop(columns=['keywords_len'])
    # Normalize the input string: remove extra spaces and convert to lowercase
    input_string = re.sub(' +', ' ', input_string.lower())

    direct_num_match = re.search(r'([>≈]\s*\d+(\.\d+)?%)', input_string)
    if direct_num_match:
        return direct_num_match.group()

    # If the input contains a percentage (no other words)
    simple_percentage_match = re.match(r'\d+(\.\d+)?%', input_string)
    if simple_percentage_match:
        found_percentage = simple_percentage_match.group()
        other_keywords_present = any(keyword in input_string for keyword in df['keywords'].str.lower())
        if not other_keywords_present:
            return found_percentage

    for _, row in df.iterrows():
        recycled_content = str(row['recycled_content']).lower()
        if recycled_content in input_string:
            return recycled_content

    for _, row in df.iterrows():
        keywords = row['keywords'].lower().split(',')
        for keyword in keywords:
            keyword = keyword.strip()

            if 'x' in keyword:
                if '.x' in keyword:
                    num_pattern = r'(0\.\d+|\.\d+|\d+)'
                else:
                    num_pattern = r'\d+'
                keyword_x_replaced = keyword.replace('.x', num_pattern)
                match = re.search(keyword_x_replaced, input_string)

                if match:
                    num_str = match.group()
                    num_match = re.search(r'\b0\.\d+\b|\.\d+|\b\d+\b', num_str)
                    if num_match:
                        num_str = num_match.group()
                        if '.' in num_str and not num_str.startswith('0'):
                            num_str = '0' + num_str
                        num = float(num_str)
                        if '.' in num_str:
                            num *= 100
                    return row['recycled_content'].replace('x', str(num))

            if keyword in input_string:
                return row['recycled_content']

    return None



def recycle_content_process(value):
    if '>' in value:
        val = '>'
        num = float(value[2:-1])
    elif '≈' in value:
        val = '='
        num = float(value[2:-1])
    else:
        val = 'x'
        num = float(value[:-1])
    return [val, num]




def complete_model_results(list_of_queries):
    tokenizer, Token_Classification_Labels, Token_Classification_model = NER_initialization()
    model_file = '../datasets_and_models/Token_Class_Model.pth'

    NER_output = Token_Classification_Results(List_of_Sentences=list_of_queries,
                                            Token_Classification_Labels=Token_Classification_Labels,
                                            Token_Classification_model=Token_Classification_model,
                                            tokenizer=tokenizer,
                                            Token_Class_File=model_file
                                            )
    if NER_output['Product Type']:
        Seq_model, le = Category_initialization('product_types')
        Seq_Class_file = '../datasets_and_models/prediction_model_product.pth'

        p_type = Category_Matching_Prediction(NER_output['Product Type'],
                 Seq_model,
                 tokenizer=tokenizer,
                 le=le,
                 Sequence_Class_File=Seq_Class_file
                     )
        NER_output['Product Type'] = p_type

    if NER_output['Material Type']:
        Seq_model, le = Category_initialization('material_types')
        Seq_Class_file = '../datasets_and_models/prediction_model_material.pth'
        m_type = Category_Matching_Prediction(NER_output['Material Type'],
                 Seq_model,
                 tokenizer=tokenizer,
                 le=le,
                 Sequence_Class_File=Seq_Class_file
                     )
        NER_output['Material Type'] = m_type

    if NER_output['Building applications']:
        Seq_model, le = Category_initialization('building_application')
        Seq_Class_file = '../datasets_and_models/prediction_building_applications.pth'
        ba_type = Category_Matching_Prediction(NER_output['Building applications'],
                 Seq_model,
                 tokenizer=tokenizer,
                 le=le,
                 Sequence_Class_File=Seq_Class_file
                     )
        NER_output['Building applications'] = ba_type

    if NER_output['Recycled Content']:
        if len(NER_output['Recycled Content']) > 1:
            NER_output['Recycled Content'] = [' '.join(NER_output['Recycled Content'])]
        NER_output['Recycled Content'] = find_recycled_content(NER_output['Recycled Content'][0])

    return NER_output




def database_search_top1(final_dict):
    df = pd.read_csv('..//datasets_and_models/dataset.csv')
    df['recycled_content'].fillna(0.0, inplace=True)
    if final_dict['Product Type']:
        words = [word[0] for word in final_dict['Product Type']]
        df = df.loc[df['product_type'].str.lower().isin([words[0]])]
    if final_dict['Material Type']:
        words = [word[0] for word in final_dict['Material Type']]
        df = df.loc[df['material_types'].str.lower().isin([words[0]])]
    if final_dict['Building applications']:
        words = [word[0] for word in final_dict['Building applications']]
        df = df.loc[df['building_application'].str.lower().isin([words[0]])]
    if final_dict['Countries']:
        df = df.loc[df['country'].str.lower().isin(final_dict['Countries'])]
    if final_dict['Recycled Content']:
        val, num = recycle_content_process(final_dict['Recycled Content'])
        if val in ['>', 'x']:
            df = df.loc[df['recycled_content'] > num]
        elif val == '=':
            df = df.loc[df['recycled_content'] == num]
    return df[['name', 'product_type', 'material_types', 'building_application', 'country', 'recycled_content']]




def database_search_top5(final_dict):
    df = pd.read_csv('..//datasets_and_models/dataset.csv')
    df['recycled_content'].fillna(0.0, inplace=True)
    if final_dict['Product Type']:
        words = [word[0] for word in final_dict['Product Type']]
        df = df.loc[df['product_type'].str.lower().isin(words)]
    if final_dict['Material Type']:
        words = [word[0] for word in final_dict['Material Type']]
        df = df.loc[df['material_types'].str.lower().isin(words)]
    if final_dict['Building applications']:
        words = [word[0] for word in final_dict['Building applications']]
        df = df.loc[df['building_application'].str.lower().isin(words)]
    if final_dict['Countries']:
        df = df.loc[df['country'].str.lower().isin(final_dict['Countries'])]
    if final_dict['Recycled Content']:
        val, num = recycle_content_process(final_dict['Recycled Content'])
        if val in ['>', 'x']:
            df = df.loc[df['recycled_content'] > num]
        elif val == '=':
            df = df.loc[df['recycled_content'] == num]
    return df[['name', 'product_type', 'material_types', 'building_application', 'country', 'recycled_content']]