from .model import Model


def generate():
    model = Model.active()
    if model:
        model.generate()


def cancel_active():
    model = Model.active()
    if model:
        model.cancel(active=True)


def cancel_queued():
    model = Model.active()
    if model:
        model.cancel(queued=True)


def cancel_all():
    model = Model.active()
    if model:
        model.cancel(active=True, queued=True)


def apply():
    model = Model.active()
    if model and model.can_apply_result:
        model.apply_current_result()
