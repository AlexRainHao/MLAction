"""
Usage: StarSpace Embedding Single-Label Classifier

An embedding intent classifier based on tensorflow
Based on `Starspace` from paper:
    `StarSpace: Embed All The Things!`
    https://arxiv.org/pdf/1709.03856.pdf

Examples:
    Train.
        >>> size = 500
        >>> dim = 100
        >>> intent = np.random.choice(list("abcd"), replace = True, size = 500)
        >>> feature = np.random.normal(size = (size, 100))

        >>> examples = [InputExample(_int, _fea) for _int, _fea in zip(intent, feature)]

        >>> model_op = StarSpaceModelPipeline(component_config = defaults)
        >>> model_op.train(examples)
        >>> model_op.persist("out")

    Predict.
        >>> model_op = StarSpaceModelPipeline().load("out")
        >>> test_case = np.random.choice(list("abcd"), size = 1)
        >>> test_feature = np.random.normal(size = (1, 100))
        >>> test_exam = InputExample(test_case, test_feature)

        >>> print(model_op.predict(test_exam))
"""

from typing import List, Dict, Union, Any, Text, Tuple

from utils import *
from model import *

from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
import json
import os
import os.path as opt
import pickle

import tensorflow as tf

STARSPACE_NAME = "start_space_model"


defaults = {
        # nn architecture
        "num_hidden_layers_a": 2,
        "hidden_layer_size_a": [256, 128],
        "num_hidden_layers_b": 0,
        "hidden_layer_size_b": [],
        "batch_size": 64,
        "epochs": 100,
        "folds": 0.8,

        # embedding parameters
        "embed_dim": 20,
        "mu_pos": 0.8,
        "mu_neg": -0.4,
        "similarity_type": "cosine",
        "num_neg": 20,
        "use_max_sim_neg": True,

        # regularization
        "C2": .002,
        "C_emb": .8,
        "dropout_rate": .8,

        # flag if tokenize intents
        "intent_split_symbol": "_",

        # visualization of accuracy
        "evaluate_every_num_epochs": 10,
        "evaluate_on_num_examples": 1000,
    }

class StarSpaceModelPipeline:
    def __init__(self,
                 component_config:Dict = None,
                 session = None,
                 graph = None,
                 idx2int: Dict = None,
                 encoded_intents_bag: np.ndarray = None,
                 text_input_dim: int = None,
                 intent_input_dim: int = None,
                 **kwargs):
        self.component_config = component_config
        self.session = session
        self.graph = graph
        self.idx2int = idx2int
        self.encoded_intents_bag = encoded_intents_bag

        self.text_input_dim = text_input_dim
        self.intent_input_dim = intent_input_dim

    @staticmethod
    def get_config_proto():
        """training session config,
        GPU would auto determined"""

        config = tf.ConfigProto(
            device_count={
                          "CPU": cpu_count(),
                          },

            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options={
                "allow_growth": True,
                "per_process_gpu_memory_fraction": 0.5,
            }
        )

        return config

    def create_vocabs(self, training_data: List[InputExample]) -> None:
        """
        create vocab for
            * intent to index vocab
            * index to intent vocab
            * intent bag vector
        """

        symbols = self.component_config.get("intent_split_symbol", "")

        intents = set([x.get("intent", "default") for x in training_data])

        self.int2idx = create_intent_dict(intents)

        if len(self.int2idx) < 2:
            raise ValueError("Intent sparse error, Need at least 2 different classes")

        self.idx2int = {val: key for key, val in self.int2idx.items()}
        self.encoded_intents_bag = create_encoded_intent_bag(self.int2idx, symbols)

    def _prepare_feature(self, training_data: List[InputExample]):
        """obtain single dataset features"""

        x_features = []
        intent_ids = []
        y_labels = []
        total_size = 0

        for exam in training_data:
            x_features.append(exam.get("feature", np.array([])))
            intent_id = self.int2idx[exam.get("intent", "default")]
            intent_ids.append(intent_id)

            y_labels.append(self.encoded_intents_bag[intent_id])

            total_size += 1


        if total_size:
            features = {
                "x": np.stack(x_features).squeeze(),
                "y": np.stack(y_labels),
                "intent": np.stack(intent_ids)
            }
        else:
            features = {
                "x": np.array([]),
                "y": np.array([]),
                "intent": np.array([])
            }

        return features, total_size

    def prepare_dataset(self, training_data: List[InputExample]):
        """obtain training dataset and dev dataset features feed into model"""

        total_size = len(training_data)

        total_pointers = np.random.permutation(total_size)

        training_size = int(total_size * self.folds)
        training_pointers = total_pointers[:training_size]
        test_pointers = total_pointers[training_size:]

        training_features, training_size = self._prepare_feature([training_data[idx] for idx in training_pointers])
        test_features, test_size = self._prepare_feature([training_data[idx] for idx in test_pointers])

        return training_features, test_features, training_size

    def eval_feature(self, feature, size):
        """obtain evaluation examples features,
        where negative sample would not used"""

        choice_index = np.random.permutation(size)

        x = feature["x"][choice_index]
        intent = feature["intent"][choice_index]

        candi_y = np.stack([self.encoded_intents_bag for _ in range(x.shape[0])])

        return {"x": x,
                "y": candi_y,
                "intent": intent}

    def negative_feature(self, feature):
        """obtain training examples equipped with negative features"""
        x = feature["x"]
        y = feature["y"]
        intent = feature["intent"]

        y = y[:, np.newaxis, :] # [batch, 1, len_int]

        neg_y = np.zeros((y.shape[0], self.component_config["num_neg"], y.shape[2]))
        for idx in range(y.shape[0]):
            neg_idx = [i for i in range(self.encoded_intents_bag.shape[0]) if i != intent[idx]]
            negs = np.random.choice(neg_idx, size = self.component_config["num_neg"])

            neg_y[idx] = self.encoded_intents_bag[negs]

        new_y = np.concatenate([y, neg_y], 1) # [batch, neg + 1, len_int]

        return {"x": x,
                "y": new_y,
                "intent": intent}

    def batcher(self, features, size, batch_size):
        """obtain batch features"""

        index = np.random.choice(size, size = size, replace = False)
        features = {"x": features["x"][index],
                    "y": features["y"][index],
                    "intent": features["intent"][index]}

        for id in range(0, size, batch_size):
            yield {"x": features["x"][id: min(id + batch_size, size)],
                   "y": features["y"][id: min(id + batch_size, size)],
                   "intent": features["intent"][id: min(id + batch_size, size)]}

    def train(self, training_data: List[InputExample]):
        self.create_vocabs(training_data)

        self.folds = min(max(self.component_config["folds"], 0.2), 1.)
        training_features, test_features, training_size = self.prepare_dataset(training_data)

        self.text_input_dim = training_features["x"].shape[-1]
        self.intent_input_dim = training_features["y"].shape[-1]

        model = StarSpace(text_dim=self.text_input_dim,
                          intent_dim=self.intent_input_dim,
                          **self.component_config)

        self.graph = model.build_graph()
        batch_size = self.component_config["batch_size"]
        epochs = self.component_config["epochs"]
        ep_visual = self.component_config["evaluate_every_num_epochs"]
        dropout = min(max(self.component_config["dropout_rate"], 0.5), 1.)

        pbar = tqdm(total=epochs, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        with self.graph.as_default():
            self.session = tf.Session(config = self.get_config_proto(),
                                      graph = self.graph)
            self.session.run(tf.global_variables_initializer())

            best_para = float("inf")
            best_step = 0
            for ep in range(epochs):

                b_loss = 0.

                for batch_data in self.batcher(training_features, training_size, batch_size):
                    # get negative features
                    batch_data = self.negative_feature(batch_data)

                    _, b_loss, b_pred = self.session.run([model.train_op, model.loss, model.pred_ids],
                                                         feed_dict = {model.text_in: batch_data["x"],
                                                                      model.intent_in: batch_data["y"],
                                                                      model.label_in: batch_data["intent"],
                                                                      model.dropout: dropout})

                # evaluate
                if ep_visual > 0:
                    if ep % ep_visual == 0:
                        train_eval_res, test_eval_res = self.evaluate(model,
                                                                      training_features,
                                                                      training_size,
                                                                      test_features)
                    else:
                        train_eval_res, test_eval_res = 0., 0.

                    pbar.set_postfix({
                        "ep": ep,
                        "loss": f"{b_loss:.3f}",
                        "tr_acc": f"{train_eval_res:.3f}",
                        "te_acc": f"{test_eval_res:.3f}"
                    })

                    _eval_para = test_eval_res or train_eval_res

                    if (-1. * _eval_para) >= (-1. * best_para):
                        best_step = ep
                        best_para = _eval_para

                else:
                    pbar.set_postfix({"ep": ep,
                                      "loss": f"{b_loss:.3f}"})
                    _eval_para = b_loss
                    if _eval_para <= best_para:
                        best_step = ep
                        best_para = _eval_para

                pbar.update(1)

            print(f"Finished training StarSpace Embedding Intent policy, "
                  f"Best training epoch at step {best_step}")

    def evaluate(self, model, train_features, training_size, test_features):
        """pass"""
        num_train_eval = min(max(self.component_config["evaluate_on_num_examples"], 1), training_size)
        train_eval_feature = self.eval_feature(train_features, num_train_eval)

        test_eval_feature = None
        test_eval_res = 0.

        if test_features["x"].size != 0:
            test_eval_feature = self.eval_feature(test_features, test_features["x"].shape[0])

        # evaluation for training samples
        train_eval_pred_ids = self.session.run(model.pred_ids,
                                               feed_dict = {model.text_in: train_eval_feature["x"],
                                                            model.intent_in: train_eval_feature["y"],
                                                            model.label_in: train_eval_feature["intent"],
                                                            model.dropout: 1.})

        train_eval_res = np.mean(train_eval_pred_ids.flatten() == train_eval_feature["intent"].flatten()).tolist()

        # evaluation for test samples
        if test_features["x"].size != 0:
            test_eval_pred_ids = self.session.run(model.pred_ids,
                                                  feed_dict={model.text_in: test_eval_feature["x"],
                                                             model.intent_in: test_eval_feature["y"],
                                                             model.label_in: test_eval_feature["intent"],
                                                             model.dropout: 1.})
            test_eval_res = np.mean(test_eval_pred_ids.flatten() == test_eval_feature["intent"].flatten()).tolist()

        return train_eval_res, test_eval_res

    def predict(self, message: InputExample, **kwargs: Any):
        """pass"""
        intent = {"name": None, "confidence": 0.}
        intent_ranking = []

        if self.graph is None:
            print("No model loaded")

        else:
            feature = {"x": message.get("feature").reshape(1, -1),
                       "y": np.stack([self.encoded_intents_bag for _ in range(1)]),
                       "intent": np.array([0])}

            sim, a, b = self.session.run([self.graph.get_tensor_by_name("sim:0"),
                                          self.graph.get_tensor_by_name("text_embed:0"),
                                          self.graph.get_tensor_by_name("intent_embed:0")],
                                         feed_dict = {self.graph.get_tensor_by_name("text_in:0"): feature["x"],
                                                      self.graph.get_tensor_by_name("intent_in:0"): feature["y"],
                                                      self.graph.get_tensor_by_name("label_in:0"): feature["intent"],
                                                      self.graph.get_tensor_by_name("dropout:0"): 1.})

            if self.component_config["similarity_type"] == "cosine":
                sim[sim < 0.] = 0.

            elif self.component_config["similarity_type"] == "inner":
                sim = np.exp(sim) / np.sum(sim)

            sim = sim.flatten()
            intent_ids = np.argsort(sim)[::-1]

            intent = {"name": self.idx2int[intent_ids[0]],
                      "confidence": sim[intent_ids[0]].tolist()}

            intent_ranking = [{"name": self.idx2int[intent_ids[idx]],
                               "confidence": sim[idx].tolist()} for idx in intent_ids]

        return json.dumps({
            "intent": intent,
            "intent_ranking": intent_ranking
        })

    @classmethod
    def load(cls,
             model_dir: Text):
        """load a model for serving from trained model config"""

        with open(opt.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)

        relative_dictionary_path = metadata.get("model_name", None)

        if model_dir and relative_dictionary_path:
            ckpt = opt.join(model_dir, relative_dictionary_path, "star_space.ckpt")
            init_text_dim = metadata["init_text_dim"]
            init_intent_dim = metadata["init_intent_dim"]

            with open(opt.join(model_dir, relative_dictionary_path, "intent_vocab.pkl"), "rb") as f:
                intent_vocab = pickle.load(f)
                idx2int = intent_vocab["idx2int"]
                encoded_intent = intent_vocab["encoded_intent"]

            graph = tf.Graph()

            with graph.as_default():
                sess = tf.Session(config=cls.get_config_proto(), graph=graph)
                saver = tf.train.import_meta_graph(ckpt + '.meta')
                saver.restore(sess, ckpt)

                return cls(metadata,
                           session=sess,
                           graph=graph,
                           idx2int=idx2int,
                           encoded_intents_bag=encoded_intent,
                           text_input_dim=init_text_dim,
                           intent_input_dim=init_intent_dim)

        else:
            return cls(metadata)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        """save trained model"""

        _arg = {"model_name": None,
                "init_text_dim": None,
                "init_intent_dim": None}

        if self.session is not None:

            target_path = opt.join(model_dir, STARSPACE_NAME)

            create_folder(target_path)

            with self.graph.as_default():
                saver = tf.train.Saver(tf.global_variables(), max_to_keep = 2)
                saver.save(self.session, opt.join(target_path, "star_space.ckpt"))

            with open(opt.join(target_path, "intent_vocab.pkl"), "wb") as f:
                pickle.dump({
                    "idx2int": self.idx2int,
                    "encoded_intent": self.encoded_intents_bag
                }, f)


            _arg = {"model_name": STARSPACE_NAME,
                    "init_text_dim": self.text_input_dim,
                    "init_intent_dim": self.intent_input_dim}

        self.component_config.update(_arg)

        with open(opt.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(self.component_config, f, ensure_ascii = False, indent = 2)