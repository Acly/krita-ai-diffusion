from ..model import Model, Workspace
from ..root import root


def generate():
    if model := root.model_for_active_document():
        if model.workspace is Workspace.generation:
            model.generate()
        elif model.workspace is Workspace.upscaling:
            model.upscale_image()


def cancel_active():
    if model := root.model_for_active_document():
        model.cancel(active=True)


def cancel_queued():
    if model := root.model_for_active_document():
        model.cancel(queued=True)


def cancel_all():
    if model := root.model_for_active_document():
        model.cancel(active=True, queued=True)


def apply():
    if model := root.model_for_active_document():
        if model.can_apply_result:
            model.apply_current_result()


def set_workspace(workspace: Workspace):
    def action():
        if model := root.model_for_active_document():
            model.workspace = workspace

    return action


def toggle_workspace():
    if model := root.model_for_active_document():
        l = list(Workspace)
        next = l[(l.index(model.workspace) + 1) % len(l)]
        model.workspace = next
