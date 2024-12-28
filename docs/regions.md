# Regions

Regions are a way to assign text prompts and reference images to specific areas
in the image.

## Introduction

When writing a prompt for an entire image the text instructions can easily become quite complex.
It's possible to describe objects in different parts of the image with phrases like
"on the left", "on the right", "in the corner" but it's not intuitive or precise.
It's also common that colors, attributes and concepts get mixed up and muddled.

Observe the following example:
> a simple wooden table in a stark empty room, on the left a white porcelain vase with light blue cornflowers, on the right an earthenware bowl, rough shape, painted with muted shades of orange and red

![img]

These are the first few outputs from SDXL. Note that colors and descriptions are all over the image. A model with better text understanding (eg. Flux) would do much better for this simple example, but runs into similar issues for longer descriptions with more subjects.

Let's try with regions:
![img]

> _Root:_ a vase with flowers and a bowl on a wooden table
> _Background (white):_ a simple wooden table in a stark empty room
> _Blue:_ a white porcelain vase with light blue cornflowers
> _Red:_ an earthenware bowl, rough shape, painted with muted shades of orange and red

The prompt is split up between three regions. Specifying "on the left", "on the right" in the text is no longer needed. Instead there is a "Root" prompt which describes the overall composition of the image without going into details.

![img]

Again, these are the first results with the same model and parameters as above. The images
aren't perfect, but subjects are cleanly separated as indicated by the coverage of the region layers.

### Composition

To exclude a frequent misunderstanding: regions are _not_ a tool for composition. They constrain
text prompts to certain regions, but do not enforce subjects to match a certain shape, or even
be generated at all!

The following setup does not lead to the desired result:
![img]

Instead, consider using a compositional tool like line or depth [Control layers]():
![img]

Alternatively build your image from the ground up with [Regions in Live mode]().


## Basics

This section covers some general advice for using regions.

### Creating regions

To create a region, use the ![icon] button next to the text input.

Regions must be attached to paint layers or layer groups. When creating a new region,
it is linked to the active layer by default. If the active layer already
has a region, creating a new region will automatically create a new layer. 

The area covered by a region is defined by the pixel opacity of the linked it is linked to.
Fully transparent pixels will not be affected.

Selecting a region's text input will also make the linked layer active, and the text focus
will follow when switching layers. Regions can be removed with the ![icon] button - this will
remove the associated text, but keep the layer. A region can also be ![icon] unlinked from the active
or ![icon] linked.

> !NOTE
> A regional prompt can be linked to multiple layers. This can be useful when there are
> several objects of the same kind spread out over the canvas which share the same attributes.
> Conversely, a layer can only be linked to a single region.

### Root prompt

The text input at the very bottom is reserved for the _Root prompt_. It is not linked to any layer.
It should describe
the general composition of the image with all its subjects. It may also contain instructions
about style, medium, quality, etc. which apply to the whole image. The Root prompt is appended to
the text of all other regions.

A good way to think about it is that image generation is a global task. To get a good composition,
every region generally needs to be aware of the whole image. You may describe a sailing ship in great
detail in its own region on one side of the image, and a fearsome Kraken in another, but for a
meaningful image to emerge the ship typically needs to be aware that the Kraken exists and vice versa.
This is accomplished by describing the scene briefly in the Root prompt.

> !NOTE
> Describing the scene in the Root prompt is less important when there is another way of
> guiding composition, such as existing canvas content or Control layers.

### Coverage and Layer stacking

The area affected by a region follows the stacking logic of paint layers. That is, the portion of
the composited image where a certain layer is visible is also what defines the coverage for
a region it might be linked to.

This means layer content is allowed to overlap, layers on top may hide parts of layers below.
Image generation always produces a single full image which matches the _visible_ areas
of the individual layers. Applying a generated result modifies the pixels on all affected
layers where they are visible - or creates new layers when using ![groups]().

### Background region

### Layer groups


## Workflows

### Regions from scratch

### Regions for existing images

### Live mode

### Selections

### Refine Region



## Advanced techniques

### Upscaling

### Edit coverage with Transparency masks

### Region hierarchies

