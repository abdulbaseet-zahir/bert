import os
import re
import torch
import tensorflow as tf


def convert_tf_to_pt_name(tensorflow_name):
    """
    Convert TensorFlow variable names to PyTorch naming convention.

    Args:
    tensorflow_name (str): The original TensorFlow variable name.

    Returns:
    str: The converted PyTorch variable name.
    """
    name_mappings = {
        "/": ".",
        "beta": "bias",
        "gamma": "weight",
        "kernel": "weight",
        "_embeddings": "_embeddings.weight",
        "seq_relationship.output_bias": "seq_relationship.bias",
        "seq_relationship.output_weights": "seq_relationship.weight",
    }

    for old, new in name_mappings.items():
        tensorflow_name = re.sub(old, new, tensorflow_name)

    return tensorflow_name


def load_tf_weights_in_bert(model, checkpoint_path):
    """
    Load TensorFlow BERT model weights into a PyTorch BERT model.

    Args:
    model (torch.nn.Module): The PyTorch model to load the weights into.
    checkpoint_path (str): The path to the TensorFlow checkpoint.

    Returns:
    torch.nn.Module: The PyTorch model with loaded weights.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    tf_model_variables = tf.train.list_variables(checkpoint_path)
    pytorch_model_state_dict = model.state_dict()

    for tf_name, shape in tf_model_variables:
        tf_tensor = torch.from_numpy(tf.train.load_variable(checkpoint_path, tf_name))
        pt_name = convert_tf_to_pt_name(tf_name)
        pt_tensor = pytorch_model_state_dict[pt_name]

        if tf_tensor.ndim == 2:
            tf_tensor = tf_tensor.T
            if tf_tensor.shape != pt_tensor.shape:
                tf_tensor = tf_tensor.T
                if tf_tensor.shape != pt_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {tf_name}: "
                        f"expected {pt_tensor.shape}, got {tf_tensor.shape}"
                    )

        pytorch_model_state_dict[pt_name].copy_(tf_tensor)

    model.load_state_dict(pytorch_model_state_dict)
    # model.eval()
    return model
