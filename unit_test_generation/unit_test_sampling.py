from sentence_transformers import SentenceTransformer
import numpy as np
import random
import gc
import torch

class TextSampler:
    def __init__(self,  
                 model_name='all-MiniLM-L6-v2', 
                 sampling_strategy='random', 
                 filter_long_answers=True):
        if 'coverage' in sampling_strategy:
            self.model = SentenceTransformer(model_name)
        self.sampling_strategy = sampling_strategy
        self.filter_long_answers = filter_long_answers
        

    def sample(self, texts, num_samples=5):
        if len(texts) <= num_samples:
            return texts
        if self.filter_long_answers:
            short_answer_texts = [t for t in texts if len(t[1].split(' '))<5]
            if len(short_answer_texts) == 0:
                pass
            else:
                texts = short_answer_texts
        orig_texts = texts
        if self.sampling_strategy == 'random':
            return random.sample(texts, min(num_samples, len(texts)))
        
        texts, answers = zip(*texts)
        if 'answer' in self.sampling_strategy:
            # Categorize texts by answers
            text_categories = {}
            for text, answer in zip(texts, answers):
                if answer not in text_categories:
                    text_categories[answer] = []
                text_categories[answer].append(text)
                
            # Select one text from each category
            selected_texts = [texts[0] for texts in text_categories.values()]
        else:
            # Select first text randomly if not considering answers
            selected_texts = [np.random.choice(texts)]
        try:
            if 'coverage' in self.sampling_strategy:
                embeddings = self.model.encode(texts)
                # Mapping from texts to their embeddings
                text_to_embedding = {text: emb for text, emb in zip(texts, embeddings)}

                # Collect embeddings of selected texts
                selected_embeddings = [text_to_embedding[text] for text in selected_texts]
                
                # Add additional selections to reach desired number while maximizing diversity
                while len(selected_texts) < num_samples:
                    remaining_texts = [text for text in texts if text not in selected_texts]
                    remaining_embeddings = [text_to_embedding[text] for text in remaining_texts]

                    # Calculate distances from the new points to already selected points
                    min_distances = np.min([np.linalg.norm(remaining_embeddings - selected_emb, axis=1) for selected_emb in selected_embeddings], axis=0)
                    new_index = np.argmax(min_distances)  # Select the point with maximum minimum distance

                    # Add the new selection
                    selected_texts.append(remaining_texts[new_index])
                    selected_embeddings.append(remaining_embeddings[new_index])
            else:
                selected_texts.extend(np.random.choice([text for text in texts if text not in selected_texts], min(num_samples - len(selected_texts), len(texts)-len(selected_texts)), replace=False))
        except:
            print("Error in sampling. Returning random samples for the rest.")
            if num_samples - len(selected_texts) > 0:
                remaining_texts = [text for text in texts if text not in selected_texts]
                if len(remaining_texts) == 0:
                    pass
                else:
                    selected_texts.extend(random.sample(remaining_texts, min(num_samples - len(selected_texts), len(remaining_texts))))
        selected_indices = [texts.index(text) for text in selected_texts]
        return [orig_texts[i] for i in selected_indices][:num_samples]

    def clear_sampler(self):
        if 'coverage' in self.sampling_strategy:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()