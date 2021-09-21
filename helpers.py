import os
import re
from collections import OrderedDict

import torch
from fuzzywuzzy import fuzz, process

import utils.unpickler as unpickler


def build_aliases(models):
    """Builds aliases for fuzzy string matching the model name input"""
    aliases = {}

    # Create aliases for models based on unique parts
    for model in models:
        name = os.path.splitext(os.path.basename(model))[0]
        parts = re.findall(r"([0-9]+x?|[A-Z]+(?![a-z])|[A-Z][^A-Z0-9_-]*)", name)
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                alias = "".join(parts[i:j])
                if alias in aliases:
                    if fuzz.ratio(alias, model) > fuzz.ratio(alias, aliases[alias]):
                        aliases[alias] = model
                else:
                    aliases[alias] = model

    # Ensure exact names are usable
    for model in models:
        name = os.path.splitext(os.path.basename(model))[0]
        aliases[name] = model

    # Build list of usable aliases
    fuzzylist = []
    for alias in aliases:
        if aliases[alias]:
            fuzzylist.append(alias)

    print("Made {} aliases for {} models.".format(len(fuzzylist), len(models)))
    return fuzzylist, aliases


def fuzzy_load_model(model_name, aliases, fuzzymodels):
    # Extract full model name from alias string
    full_model_name = aliases[
        process.extractOne(model_name.replace(".pth", ""), fuzzymodels)[0]
    ]
    # Load the model in torch using safe unpickler
    model = torch.load(
        os.path.join("./models/", full_model_name),
        pickle_module=unpickler.RestrictedUnpickle,
    )
    return full_model_name, model


def interpolate_models(models, amounts):
    # Normalize to 0-1 range
    amounts = [int(amount) / sum(amounts) for amount in amounts]
    # Initialize an empty state_dict to store model info in
    state_dict = OrderedDict()
    # Initialize each entry in the state dict based on first model
    for k, _ in models[0].items():
        state_dict[k] = 0
    for idx, model in enumerate(models):
        for k, _ in models[idx].items():
            # Grab model entry value
            v_n = model[k]
            # Scale value by amount and add to state_dict entry
            state_dict[k] += v_n * amounts[idx]
    return state_dict


def parse_models(model_string, aliases, fuzzymodels):
    jobs = []
    model_jobs = model_string.split(";")
    for model_job in model_jobs:
        model_chain = model_job.split(">")
        chain = []
        for chained_model in model_chain:
            interpolations = chained_model.split("&")
            if len(interpolations) > 1:
                loaded_models = []
                amounts = []
                model_names = []
                for interp in interpolations:
                    try:
                        model_name, amount = interp.split(":")
                        full_model_name, loaded_model = fuzzy_load_model(
                            model_name, aliases, fuzzymodels
                        )
                        model_names.append(full_model_name)
                        loaded_models.append(loaded_model)
                        amounts.append(int(amount))
                    except:
                        raise ValueError("Error parsing interpolations")
                chained_model = {
                    "model_name": "&".join(model_names),
                    "state_dict": interpolate_models(loaded_models, amounts),
                }
            else:
                full_model_name, state_dict = fuzzy_load_model(
                    chained_model, aliases, fuzzymodels
                )
                chained_model = {
                    "model_name": full_model_name,
                    "state_dict": state_dict,
                }
            chain.append(chained_model)
        jobs.append(chain)
    return jobs


models = []
for (dirpath, dirnames, filenames) in os.walk("./models"):
    models.extend(filenames)
    break
fuzzymodels, aliases = build_aliases(models)
