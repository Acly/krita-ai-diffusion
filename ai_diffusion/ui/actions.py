from .model import Model, Workspace


def generate():
    model = Model.active()
    if model and model.workspace is Workspace.generation:
        model.generate()
    elif model and model.workspace is Workspace.upscaling:
        model.upscale_image()


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


def set_workspace(workspace):
    def action():
        model = Model.active()
        if model:
            if model.workspace is Workspace.live:
                model.live.is_active = False
            model.workspace = workspace
            model.changed.emit()

    return action


def toggle_workspace():
    model = Model.active()
    if model:
        if model.workspace is Workspace.live:
            model.live.is_active = False
        l = list(Workspace)
        next = l[(l.index(model.workspace) + 1) % len(l)]
        model.workspace = next
        model.changed.emit()
