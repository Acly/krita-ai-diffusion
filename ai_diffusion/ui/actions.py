from ..model import Workspace
from ..root import root


def generate():
    if model := root.model_for_active_document():
        if model.workspace is Workspace.generation:
            model.generate()
        elif model.workspace is Workspace.upscaling:
            model.upscale_image()
        elif model.workspace is Workspace.live:
            model.generate_live()
        elif model.workspace is Workspace.animation:
            model.animation.generate()
        elif model.workspace is Workspace.custom:
            model.custom.generate()


def cancel_active():
    if model := root.model_for_active_document():
        model.cancel(active=True)


def cancel_queued():
    if model := root.model_for_active_document():
        model.cancel(queued=True)


def cancel_all():
    if model := root.model_for_active_document():
        model.cancel(active=True, queued=True)


def toggle_preview():
    if model := root.model_for_active_document():
        model.jobs.toggle_selection()


def apply():
    if model := root.model_for_active_document():
        if model.workspace is Workspace.generation and len(model.jobs.selection) > 0:
            model.apply_generated_result(*model.jobs.selection[0])
        elif model.workspace is Workspace.live:
            model.live.apply_result()


def apply_alternative():
    if model := root.model_for_active_document():
        if model.workspace is Workspace.live:
            model.live.apply_result(layer_only=True)
        else:
            apply()


def create_region():
    if model := root.model_for_active_document():
        model.regions.create_region(group=model.workspace is not Workspace.live)


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
