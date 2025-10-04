from pymongo import MongoClient
from numpy.linalg import norm
import numpy as np
import os
import requests
import re

# Helps in comparison
def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (norm(a) * norm(b))

class RAGDataBase:

    # Database address and name
    def __init__(self,uri = 'mongodb://127.0.0.1:27017', db_name = 'rag_nasa_database'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.questions = self.db['questions']
        self.images = self.db['images']
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.IMG_DIR = os.path.join(self.PROJECT_ROOT, 'images')

        self.last_id_file = os.path.join(self.IMG_DIR, f'available_id.txt')
        self.id_image = 0
        with open(self.last_id_file, 'r') as f:
           self.id_image = int(f.read())

        os.makedirs(self.IMG_DIR, exist_ok=True)
        self.pattern = re.compile(r"\.(png|jpe?g|gif|bmp|webp|svg|tiff?)$", re.IGNORECASE)

        # Deletion types
        self.delete_one = 0
        self.delete_many = 1
        self.delete_all = 2

    def insert_question(self,question, answer, embeddings= None,image_ids = None):
        doc = {
            'question': question,
            'answer': answer,
            'embeddings': embeddings.tolist() if embeddings is not None else None,
            'image_ids' : image_ids
        }
        self.questions.insert_one(doc)

    def get_all_questions(self):
        return list(self.questions.find({},{'_id':0}))

    def get_all_images(self):
        return list(self.images.find({},{'_id':0}))


    def search_question(self,question_embeddings,threshold = 0.8):
        if question_embeddings is None:
            return None, None, None

        docs = self.get_all_questions()
        if not docs:
            return None, None, None

        embeddings = np.array([doc['embeddings'] for doc in docs])

        # Checks similarity between question embedding and the embeddings of all the questions in the Database
        similarities = [cosine_similarity(question_embeddings,e) for e in embeddings]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Gets best match if the similarity is above the threshold
        if best_score >= threshold:
            best_match = docs[best_idx]
            return best_match['question'], best_match['answer'], best_match['image_ids']
        return None,None , None

    def delete(self,collection,embedded_question=None,deletion_type=0):

        if deletion_type == self.delete_all:
            if collection == 'questions':
                self.questions.delete_many({})
            elif collection == 'images':
                self.images.delete_many({})
                with open(self.last_id_file, 'w') as f:
                    f.write(str(0))
                self.id_image = 0
            return True
        question, answer, images = self.search_question(embedded_question)

        if question:
            match deletion_type:
                case self.delete_one:
                    self.questions.delete_one({collection: {'$regex': question, '$options': 'i'}})
                    return True
                case self.delete_many:
                    self.questions.delete_many({collection: {'$regex': question, '$options': 'i'}})
                    return True
        return False

    def image_type(self,url):
        match = self.pattern.search(url)
        if match:
            return match.group(1)
        return None

    def download_image(self,url):
        img_path = os.path.join(self.IMG_DIR, f'{self.id_image}.{self.image_type(url)}')

        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(r.content)
            return img_path
        else:
            raise Exception(f'Failed to download {url}, status {r.status_code}')

    def insert_image(self,url):

        doc = {
            'id' : self.id_image,
            'url' : url,
            'location' : self.download_image(url)
        }
        self.id_image += 1
        with open(self.last_id_file, 'w') as f:
            f.write(str(self.id_image))
        self.images.insert_one(doc)
        return f'<[{int(self.id_image) - 1}]>'