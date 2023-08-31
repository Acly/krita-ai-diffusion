from .model import Model


def generate():
    model = Model.active()
    if model:
        model.generate()


def cancel():
    model = Model.active()
    if model and model.jobs.any_executing():
        model.cancel()


def apply():
    model = Model.active()
    if model and model.can_apply_result:
        model.apply_current_result()
