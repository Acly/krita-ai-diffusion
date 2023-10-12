from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from typing import List, Dict, Optional, Union

KisPresetChooser = QObject
KoDockFactoryBase = QObject

class Window(QObject):
    """* Window represents one Krita mainwindow. A window can have any number of views open on any number of documents."""

    def qwindow(self) -> QMainWindow:
        # type: () -> QMainWindow:
        """@access public Q_SLOTS
        Return a handle to the QMainWindow widget. This is useful to e.g. parent dialog boxes and message box.
        """
    def dockers(self) -> List[QDockWidget]:
        # type: () -> List[QDockWidget]:
        """@access public Q_SLOTS
         @brief dockers
        @return a list of all the dockers belonging to this window"""
    def views(self) -> List["View"]:
        # type: () -> List[View]:
        """@access public Q_SLOTS
        @return a list of open views in this window"""
    def addView(self, document: "Document") -> "View":
        # type: (document) -> View:
        """@access public Q_SLOTS
        Open a new view on the given document in this window"""
    def showView(self, view: "View") -> None:
        # type: (view) -> None:
        """@access public Q_SLOTS
        Make the given view active in this window. If the view does not belong to this window, nothing happens.
        """
    def activeView(self) -> "View":
        # type: () -> View:
        """@access public Q_SLOTS
        @return the currently active view or 0 if no view is active"""
    def activate(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief activate activates this Window."""
    def close(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief close the active window and all its Views. If there are no Views left for a given Document, that Document will also be closed.
        """
    def createAction(
        self, id: str, text: str = str(), menuLocation: str = str("tools/scripts")
    ) -> QAction:
        # type: (id, text, menuLocation) -> QAction:
        """@access public Q_SLOTS
         @brief createAction creates a QAction object and adds it to the action manager for this Window.
        @param id The unique id for the action. This will be used to     propertize the action if any .action file is present
        @param text The user-visible text of the action. If empty, the text from the    .action file is used.
        @param menuLocation a /-separated string that describes which menu the action should     be places in. Default is "tools/scripts"
        @return the new action."""
    def windowClosed(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Emitted when the window is closed."""
    def themeChanged(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Emitted when we change the color theme"""
    def activeViewChanged(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Emitted when the active view changes"""

class View(QObject):
    """* View represents one view on a document. A document can be shown in more than one view at a time."""

    def window(self) -> "Window":
        # type: () -> Window:
        """@access public Q_SLOTS
        @return the window this view is shown in."""
    def document(self) -> "Document":
        # type: () -> Document:
        """@access public Q_SLOTS
        @return the document this view is showing."""
    def setDocument(self, document: "Document") -> None:
        # type: (document) -> None:
        """@access public Q_SLOTS
        Reset the view to show @p document."""
    def visible(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @return true if the current view is visible, false if not."""
    def setVisible(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Make the current view visible."""
    def canvas(self) -> "Canvas":
        # type: () -> Canvas:
        """@access public Q_SLOTS
        @return the canvas this view is showing. The canvas controls things like zoom and rotation.
        """
    def activateResource(self, resource: "Resource") -> None:
        # type: (resource) -> None:
        """@access public Q_SLOTS
         @brief activateResource activates the given resource.
        @param resource: a pattern, gradient or paintop preset"""
    def foregroundColor(self) -> "ManagedColor":
        # type: () -> ManagedColor:
        """@access public Q_SLOTS
         @brief foregroundColor allows access to the currently active color. This is nominally per canvas/view, but in practice per mainwindow.
        @code
        color = Application.activeWindow().activeView().foregroundColor()
        components = color.components()
        components[0] = 1.0
        components[1] = 0.6
        components[2] = 0.7
        color.setComponents(components)
        Application.activeWindow().activeView().setForeGroundColor(color)
        @endcode"""
    def showFloatingMessage(self, message: str, icon: QIcon, timeout: int, priority: int) -> None:
        # type: (message, icon, timeout, priority) -> None:
        """@access public Q_SLOTS
         @brief showFloatingMessage displays a floating message box on the top-left corner of the canvas
        @param message: Message to be displayed inside the floating message box
        @param icon: Icon to be displayed inside the message box next to the message string
        @param timeout: Milliseconds until the message box disappears
        @param priority: 0 = High, 1 = Medium, 2 = Low. Higher priority messages will be displayed in place of lower priority messages
        """
    def selectedNodes(self) -> List["Node"]:
        # type: () -> List[Node]:
        """@access public Q_SLOTS
         @brief selectedNodes returns a list of Nodes that are selected in this view.
        @code
        from krita import *
        w = Krita.instance().activeWindow()
        v = w.activeView()
        selected_nodes = v.selectedNodes()
        print(selected_nodes)
        @endcode
        @return a list of Node objects which may be empty."""
    def flakeToDocumentTransform(self) -> QTransform:
        # type: () -> QTransform:
        """@access public Q_SLOTS
         @brief flakeToDocumentTransform The tranformation of the document relative to the view without rotation and mirroring
        @return QTransform"""
    def flakeToCanvasTransform(self) -> QTransform:
        # type: () -> QTransform:
        """@access public Q_SLOTS
         @brief flakeToCanvasTransform The tranformation of the canvas relative to the view without rotation and mirroring
        @return QTransform"""
    def flakeToImageTransform(self) -> QTransform:
        # type: () -> QTransform:
        """@access public Q_SLOTS
         @brief flakeToImageTransform The tranformation of the image relative to the view without rotation and mirroring
        @return QTransform"""

class Shape(QObject):
    """* @brief The Shape class The shape class is a wrapper around Krita's vector objects. Some example code to parse through interesting information in a given vector layer with shapes.
    @code
    import sys
    from krita import *

    doc = Application.activeDocument()

    root = doc.rootNode()

    for layer in root.childNodes():
        print (str(layer.type())+" "+str(layer.name()))
        if (str(layer.type())=="vectorlayer"):
            for shape in layer.shapes():
                print(shape.name())
                print(shape.toSvg())
    @endcode"""

    def name(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief name
        @return the name of the shape"""
    def setName(self, name: str) -> None:
        # type: (name) -> None:
        """@access public Q_SLOTS
         @brief setName
        @param name which name the shape should have."""
    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type
        @return the type of shape."""
    def zIndex(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief zIndex
        @return the zindex of the shape."""
    def setZIndex(self, zindex: int) -> None:
        # type: (zindex) -> None:
        """@access public Q_SLOTS
         @brief setZIndex
        @param zindex set the shape zindex value."""
    def selectable(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief selectable
        @return whether the shape is user selectable."""
    def setSelectable(self, selectable: bool) -> None:
        # type: (selectable) -> None:
        """@access public Q_SLOTS
         @brief setSelectable
        @param selectable whether the shape should be user selectable."""
    def geometryProtected(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief geometryProtected
        @return whether the shape is protected from user changing the shape geometry."""
    def setGeometryProtected(self, protect: bool) -> None:
        # type: (protect) -> None:
        """@access public Q_SLOTS
         @brief setGeometryProtected
        @param protect whether the shape should be geometry protected from the user."""
    def visible(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief visible
        @return whether the shape is visible."""
    def setVisible(self, visible: bool) -> None:
        # type: (visible) -> None:
        """@access public Q_SLOTS
         @brief setVisible
        @param visible whether the shape should be visible."""
    def boundingBox(self) -> QRectF:
        # type: () -> QRectF:
        """@access public Q_SLOTS
         @brief boundingBox the bounding box of the shape in points
        @return RectF containing the bounding box."""
    def position(self) -> QPointF:
        # type: () -> QPointF:
        """@access public Q_SLOTS
         @brief position the position of the shape in points.
        @return the position of the shape in points."""
    def setPosition(self, point: QPointF) -> None:
        # type: (point) -> None:
        """@access public Q_SLOTS
         @brief setPosition set the position of the shape.
        @param point the new position in points"""
    def transformation(self) -> QTransform:
        # type: () -> QTransform:
        """@access public Q_SLOTS
         @brief transformation the 2D transformation matrix of the shape.
        @return the 2D transformation matrix."""
    def setTransformation(self, matrix: QTransform) -> None:
        # type: (matrix) -> None:
        """@access public Q_SLOTS
         @brief setTransformation set the 2D transformation matrix of the shape.
        @param matrix the new 2D transformation matrix."""
    def absoluteTransformation(self) -> QTransform:
        # type: () -> QTransform:
        """@access public Q_SLOTS
         @brief transformation the 2D transformation matrix of the shape including all grandparent transforms.
        @return the 2D transformation matrix."""
    def remove(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @brief remove delete the shape."""
    def update(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief update queue the shape update."""
    def updateAbsolute(self, box: QRectF) -> None:
        # type: (box) -> None:
        """@access public Q_SLOTS
         @brief updateAbsolute queue the shape update in the specified rectangle.
        @param box the RectF rectangle to update."""
    def toSvg(self, prependStyles: bool = False, stripTextMode: bool = True) -> str:
        # type: (prependStyles, stripTextMode) -> str:
        """@access public Q_SLOTS
         @brief toSvg convert the shape to svg, will not include style definitions.
        @param prependStyles prepend the style data. Default: false
        @param stripTextMode enable strip text mode. Default: true
        @return the svg in a string."""
    def select(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief select selects the shape."""
    def deselect(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief deselect deselects the shape."""
    def isSelected(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief isSelected
        @return whether the shape is selected."""
    def parentShape(self) -> "Shape":
        # type: () -> Shape:
        """@access public Q_SLOTS
         @brief parentShape
        @return the parent GroupShape of the current shape."""

class Selection(QObject):
    """* Selection represents a selection on Krita. A selection is not necessarily associated with a particular Node or Image.
    @code
    from krita import *
    d = Application.activeDocument()
    n = d.activeNode()
    r = n.bounds()
    s = Selection()
    s.select(r.width() / 3, r.height() / 3, r.width() / 3, r.height() / 3, 255)
    s.cut(n)
    @endcode"""

    def __init__(self) -> "Shape":
        """@brief parentShape
        @return the parent GroupShape of the current shape."""
    def duplicate(self) -> "Selection":
        # type: () -> Selection:
        """@access public Q_SLOTS
        @return a duplicate of the selection"""
    def width(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the width of the selection"""
    def height(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the height of the selection"""
    def x(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the left-hand position of the selection."""
    def y(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the top position of the selection."""
    def move(self, x: int, y: int) -> None:
        # type: (x, y) -> None:
        """@access public Q_SLOTS
        Move the selection's top-left corner to the given coordinates."""
    def clear(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Make the selection entirely unselected."""
    def contract(self, value: int) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        Make the selection's width and height smaller by the given value. This will not move the selection's top-left position.
        """
    def copy(self, node: "Node") -> None:
        # type: (node) -> None:
        """@access public Q_SLOTS
         @brief copy copies the area defined by the selection from the node to the clipboard.
        @param node the node from where the pixels will be copied."""
    def cut(self, node: "Node") -> None:
        # type: (node) -> None:
        """@access public Q_SLOTS
         @brief cut erases the area defined by the selection from the node and puts a copy on the clipboard.
        @param node the node from which the selection will be cut."""
    def paste(self, destination: "Node", x: int, y: int) -> None:
        # type: (destination, x, y) -> None:
        """@access public Q_SLOTS
         @brief paste pastes the content of the clipboard to the given node, limited by the area of the current selection.
        @param destination the node where the pixels will be written
        @param x: the x position at which the clip will be written
        @param y: the y position at which the clip will be written"""
    def erode(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Erode the selection with a radius of 1 pixel."""
    def dilate(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Dilate the selection with a radius of 1 pixel."""
    def border(self, xRadius: int, yRadius: int) -> None:
        # type: (xRadius, yRadius) -> None:
        """@access public Q_SLOTS
        Border the selection with the given radius."""
    def feather(self, radius: int) -> None:
        # type: (radius) -> None:
        """@access public Q_SLOTS
        Feather the selection with the given radius."""
    def grow(self, xradius: int, yradius: int) -> None:
        # type: (xradius, yradius) -> None:
        """@access public Q_SLOTS
        Grow the selection with the given radius."""
    def shrink(self, xRadius: int, yRadius: int, edgeLock: bool) -> None:
        # type: (xRadius, yRadius, edgeLock) -> None:
        """@access public Q_SLOTS
        Shrink the selection with the given radius."""
    def smooth(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Smooth the selection."""
    def invert(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Invert the selection."""
    def resize(self, w: int, h: int) -> None:
        # type: (w, h) -> None:
        """@access public Q_SLOTS
        Resize the selection to the given width and height. The top-left position will not be moved.
        """
    def select(self, x: int, y: int, w: int, h: int, value: int) -> None:
        # type: (x, y, w, h, value) -> None:
        """@access public Q_SLOTS
        Select the given area. The value can be between 0 and 255; 0 is  totally unselected, 255 is totally selected.
        """
    def selectAll(self, node: "Node", value: int) -> None:
        # type: (node, value) -> None:
        """@access public Q_SLOTS
        Select all pixels in the given node. The value can be between 0 and 255; 0 is  totally unselected, 255 is totally selected.
        """
    def replace(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS
        Replace the current selection's selection with the one of the given selection."""
    def add(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS
        Add the given selection's selected pixels to the current selection."""
    def subtract(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS
        Subtract the given selection's selected pixels from the current selection."""
    def intersect(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS
        Intersect the given selection with this selection."""
    def symmetricdifference(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS
        Intersect with the inverse of the given selection with this selection."""
    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        # type: (x, y, w, h) -> QByteArray:
        """@access public Q_SLOTS
         @brief pixelData reads the given rectangle from the Selection's mask and returns it as a byte array. The pixel data starts top-left, and is ordered row-first. The byte array will contain one byte for every pixel, representing the selectedness. 0 is totally unselected, 255 is fully selected. You can read outside the Selection's boundaries; those pixels will be unselected. The byte array is a copy of the original selection data.
        @param x x position from where to start reading
        @param y y position from where to start reading
        @param w row length to read
        @param h number of rows to read
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def setPixelData(
        self, value: Union[QByteArray, bytes, bytearray], x: int, y: int, w: int, h: int
    ) -> None:
        # type: (value, x, y, w, h) -> None:
        """@access public Q_SLOTS
         @brief setPixelData writes the given bytes, of which there must be enough, into the Selection.
        @param value the byte array representing the pixels. There must be enough bytes available. Krita will take the raw pointer from the QByteArray and start reading, not stopping before (w * h) bytes are read.
        @param x the x position to start writing from
        @param y the y position to start writing from
        @param w the width of each row
        @param h the number of rows to write"""

class Resource(QObject):
    """* A Resource represents a gradient, pattern, brush tip, brush preset, palette or  workspace definition.
    @code
    allPresets = Application.resources("preset")
    for preset in allPresets:
        print(preset.name())
    @endcode  Resources are identified by their type, name and filename. If you want to change the contents of a resource, you should read its data using data(), parse it and write the changed contents back.
    """

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        Return the type of this resource. Valid types are: <ul> <li>pattern: a raster image representing a pattern <li>gradient: a gradient <li>brush: a brush tip <li>preset: a brush preset <li>palette: a color set <li>workspace: a workspace definition. </ul>
        """
    def name(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        The user-visible name of the resource."""
    def setName(self, value: str) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        setName changes the user-visible name of the current resource."""
    def filename(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        The filename of the resource, if present. Not all resources are loaded from files."""
    def image(self) -> QImage:
        # type: () -> QImage:
        """@access public Q_SLOTS
        An image that can be used to represent the resource in the user interface. For some resources, like patterns, the  image is identical to the resource, for others it's a mere icon.
        """
    def setImage(self, image: QImage) -> None:
        # type: (image) -> None:
        """@access public Q_SLOTS
        Change the image for this resource."""

class Preset(QObject):
    """* @brief The Preset class Preset is a resource object that stores brush preset data. An example for printing the current brush preset and all its settings:
    @code
    from krita import *

    view = Krita.instance().activeWindow().activeView()
    preset = Preset(view.currentBrushPreset())

    print ( preset.toXML() )
    @endcode"""

    def toXML(self) -> str:
        # type: () -> str:
        """@access public
         @brief toXML convert the preset settings into a preset formatted xml.
        @return the xml in a string."""
    def fromXML(self, xml: str) -> None:
        # type: (xml) -> None:
        """@access public
         @brief fromXML convert the preset settings into a preset formatted xml.
        @param xml valid xml preset string."""
    def paintOpPreset(self) -> "KisPaintOpPresetSP":
        # type: () -> 'KisPaintOpPresetSP':
        """@access private
         @brief paintOpPreset
        @return gives a KisPaintOpPreset object back"""

class Palette(QObject):
    """* @brief The Palette class Palette is a resource object that stores organised color data. It's purpose is to allow artists to save colors and store them. An example for printing all the palettes and the entries:
    @code
    import sys
    from krita import *

    resources = Application.resources("palette")

    for (k, v) in resources.items():
        print(k)
        palette = Palette(v)
        for x in range(palette.numberOfEntries()):
            entry = palette.colorSetEntryByIndex(x)
            c = palette.colorForEntry(entry);
            print(x, entry.name(), entry.id(), entry.spotColor(), c.toQString())
    @endcode"""

    def numberOfEntries(self) -> int:
        # type: () -> int:
        """@access public
         @brief numberOfEntries
        @return"""
    def columnCount(self) -> int:
        # type: () -> int:
        """@access public
         @brief columnCount
        @return the amount of columns this palette is set to use."""
    def setColumnCount(self, columns: int) -> None:
        # type: (columns) -> None:
        """@access public
        @brief setColumnCount Set the amount of columns this palette should use."""
    def comment(self) -> str:
        # type: () -> str:
        """@access public
         @brief comment
        @return the comment or description associated with the palette."""
    def setComment(self, comment: str) -> None:
        # type: (comment) -> None:
        """@access public
         @brief setComment set the comment or description associated with the palette.
        @param comment"""
    def groupNames(self) -> List[str]:
        # type: () -> List[str]:
        """@access public
         @brief groupNames
        @return the list of group names. This is list is in the order these groups are in the file.
        """
    def addGroup(self, name: str) -> bool:
        # type: (name) -> bool:
        """@access public
         @brief addGroup
        @param name of the new group
        @return whether adding the group was successful."""
    def removeGroup(self, name: str, keepColors: bool = True) -> bool:
        # type: (name, keepColors) -> bool:
        """@access public
         @brief removeGroup
        @param name the name of the group to remove.
        @param keepColors whether or not to delete all the colors inside, or to move them to the default group.
        @return"""
    def colorsCountTotal(self) -> int:
        # type: () -> int:
        """@access public
         @brief colorsCountTotal
        @return the total amount of entries in the whole group"""
    def colorSetEntryByIndex(self, index: int) -> "Swatch":
        # type: (index) -> Swatch:
        """@access public
         @brief colorSetEntryByIndex get the colorsetEntry from the global index.
        @param index the global index
        @return the colorset entry"""
    def colorSetEntryFromGroup(self, index: int, groupName: str) -> "Swatch":
        # type: (index, groupName) -> Swatch:
        """@access public
         @brief colorSetEntryFromGroup
        @param index index in the group.
        @param groupName the name of the group to get the color from.
        @return the colorsetentry."""
    def addEntry(self, entry: "Swatch", groupName: str = str()) -> None:
        # type: (entry, groupName) -> None:
        """@access public
         @brief addEntry add an entry to a group. Gets appended to the end.
        @param entry the entry
        @param groupName the name of the group to add to."""
    def removeEntry(self, index: int, groupName: str) -> None:
        # type: (index, groupName) -> None:
        """@access public
        @brief removeEntry remove the entry at @p index from the group @p groupName."""
    def changeGroupName(self, oldGroupName: str, newGroupName: str) -> bool:
        # type: (oldGroupName, newGroupName) -> bool:
        """@access public
         @brief changeGroupName change the group name.
        @param oldGroupName the old groupname to change.
        @param newGroupName the new name to change it into.
        @return whether successful. Reasons for failure include not knowing have oldGroupName"""
    def moveGroup(self, groupName: str, groupNameInsertBefore: str = str()) -> bool:
        # type: (groupName, groupNameInsertBefore) -> bool:
        """@access public
         @brief moveGroup move the group to before groupNameInsertBefore.
        @param groupName group to move.
        @param groupNameInsertBefore group to inset before.
        @return whether successful. Reasons for failure include either group not existing."""
    def save(self) -> bool:
        # type: () -> bool:
        """@access public
         @brief save save the palette
        @return whether it was successful."""
    def colorSet(self) -> "KoColorSetSP":
        # type: () -> 'KoColorSetSP':
        """@access private
         @brief colorSet
        @return gives qa KoColorSet object back"""

class Notifier(QObject):
    """* The Notifier can be used to be informed of state changes in the Krita application."""

    def active(self) -> bool:
        # type: () -> bool:
        """@access public
        @return true if the Notifier is active."""
    def setActive(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public
        Enable or disable the Notifier"""
    def applicationClosing(self) -> None:
        # type: () -> None:
        """@access public
        @brief applicationClosing is emitted when the application is about to close. This happens after any documents and windows are closed.
        """
    def imageCreated(self, image: "Document") -> None:
        # type: (image) -> None:
        """@access public
        @brief imageCreated is emitted whenever a new image is created and registered with the application.
        """
    def imageSaved(self, filename: str) -> None:
        # type: (filename) -> None:
        """@access public
         @brief imageSaved is emitted whenever a document is saved.
        @param filename the filename of the document that has been saved."""
    def imageClosed(self, filename: str) -> None:
        # type: (filename) -> None:
        """@access public
         @brief imageClosed is emitted whenever the last view on an image is closed. The image does not exist anymore in Krita
        @param filename the filename of the image."""
    def viewCreated(self, view: "View") -> None:
        # type: (view) -> None:
        """@access public
         @brief viewCreated is emitted whenever a new view is created.
        @param view the view"""
    def viewClosed(self, view: "View") -> None:
        # type: (view) -> None:
        """@access public
         @brief viewClosed is emitted whenever a view is closed
        @param view the view"""
    def windowIsBeingCreated(self, window: "Window") -> None:
        # type: (window) -> None:
        """@access public
         @brief windowCreated is emitted whenever a window is being created
        @param window the window; this is called from the constructor of the window, before the xmlgui file is loaded
        """
    def windowCreated(self) -> None:
        # type: () -> None:
        """@access public
        @brief windowIsCreated is emitted after main window is completely created"""
    def configurationChanged(self) -> None:
        # type: () -> None:
        """@access public
        @brief configurationChanged is emitted every time Krita's configuration has changed."""

class Node(QObject):
    """* Node represents a layer or mask in a Krita image's Node hierarchy. Group layers can contain other layers and masks; layers can contain masks."""

    def __init__(self) -> None:
        """@brief configurationChanged is emitted every time Krita's configuration has changed."""
    def clone(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
        @brief clone clone the current node. The node is not associated with any image."""
    def alphaLocked(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief alphaLocked checks whether the node is a paint layer and returns whether it is alpha locked
        @return whether the paint layer is alpha locked, or false if the node is not a paint layer
        """
    def setAlphaLocked(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setAlphaLocked set the layer to value if the node is paint layer."""
    def blendingMode(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the blending mode of the layer. The values of the blending modes are defined in @see KoCompositeOpRegistry.h
        """
    def setBlendingMode(self, value: str) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setBlendingMode set the blending mode of the node to the given value
        @param value one of the string values from @see KoCompositeOpRegistry.h"""
    def channels(self) -> List["Channel"]:
        # type: () -> List[Channel]:
        """@access public Q_SLOTS
         @brief channels creates a list of Channel objects that can be used individually to show or hide certain channels, and to retrieve the contents of each channel in a node separately. Only layers have channels, masks do not, and calling channels on a Node that is a mask will return an empty list.
        @return the list of channels ordered in by position of the channels in pixel position"""
    def childNodes(self) -> List["Node"]:
        # type: () -> List[Node]:
        """@access public Q_SLOTS
         @brief childNodes
        @return returns a list of child nodes of the current node. The nodes are ordered from the bottommost up. The function is not recursive.
        """
    def findChildNodes(
        self,
        name: str = str(),
        recursive: bool = False,
        partialMatch: bool = False,
        type: str = str(),
        colorLabelIndex: int = 0,
    ) -> List["Node"]:
        # type: (name, recursive, partialMatch, type, colorLabelIndex) -> List[Node]:
        """@access public Q_SLOTS
         @brief findChildNodes
        @param name name of the child node to search for. Leaving this blank will return all nodes.
        @param recursive whether or not to search recursively. Defaults to false.
        @param partialMatch return if the name partially contains the string (case insensative). Defaults to false.
        @param type filter returned nodes based on type
        @param colorLabelIndex filter returned nodes based on color label index
        @return returns a list of child nodes and grand child nodes of the current node that match the search criteria.
        """
    def addChildNode(self, child: "Node", above: "Optional[Node]") -> bool:
        # type: (child, above) -> bool:
        """@access public Q_SLOTS
         @brief addChildNode adds the given node in the list of children.
        @param child the node to be added
        @param above the node above which this node will be placed
        @return false if adding the node failed"""
    def removeChildNode(self, child: "Node") -> bool:
        # type: (child) -> bool:
        """@access public Q_SLOTS
         @brief removeChildNode removes the given node from the list of children.
        @param child the node to be removed"""
    def setChildNodes(self, nodes: List["Node"]) -> None:
        # type: (nodes) -> None:
        """@access public Q_SLOTS
         @brief setChildNodes this replaces the existing set of child nodes with the new set.
        @param nodes The list of nodes that will become children, bottom-up -- the first node, is the bottom-most node in the stack.
        """
    def colorDepth(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @return the color depth."""
    def colorModel(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief colorModel retrieve the current color model of this document: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @return the internal color model string."""
    def colorProfile(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the name of the current color profile"""
    def setColorProfile(self, colorProfile: str) -> bool:
        # type: (colorProfile) -> bool:
        """@access public Q_SLOTS
         @brief setColorProfile set the color profile of the image to the given profile. The profile has to be registered with krita and be compatible with the current color model and depth; the image data is <i>not</i> converted.
        @param colorProfile
        @return if assigning the color profile worked"""
    def setColorSpace(self, colorModel: str, colorDepth: str, colorProfile: str) -> bool:
        # type: (colorModel, colorDepth, colorProfile) -> bool:
        """@access public Q_SLOTS
         @brief setColorSpace convert the node to the given colorspace
        @param colorModel A string describing the color model of the node: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @param colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @param colorProfile a valid color profile for this color model and color depth combination.
        """
    def animated(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief Krita layers can be animated, i.e., have frames.
        @return return true if the layer has frames. Currently, the scripting framework does not give access to the animation features.
        """
    def enableAnimation(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief enableAnimation make the current layer animated, so it can have frames."""
    def setPinnedToTimeline(self, pinned: bool) -> None:
        # type: (pinned) -> None:
        """@access public Q_SLOTS
        @brief Sets whether or not node should be pinned to the Timeline Docker, regardless of selection activity.
        """
    def isPinnedToTimeline(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @return Returns true if node is pinned to the Timeline Docker or false if it is not."""
    def setCollapsed(self, collapsed: bool) -> None:
        # type: (collapsed) -> None:
        """@access public Q_SLOTS
        Sets the state of the node to the value of @param collapsed"""
    def collapsed(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        returns the collapsed state of this node"""
    def colorLabel(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        Sets a color label index associated to the layer.  The actual color of the label and the number of available colors is defined by Krita GUI configuration.
        """
    def setColorLabel(self, index: int) -> None:
        # type: (index) -> None:
        """@access public Q_SLOTS
         @brief setColorLabel sets a color label index associated to the layer.  The actual color of the label and the number of available colors is defined by Krita GUI configuration.
        @param index an integer corresponding to the set of available color labels."""
    def inheritAlpha(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief inheritAlpha checks whether this node has the inherits alpha flag set
        @return true if the Inherit Alpha is set"""
    def setInheritAlpha(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        set the Inherit Alpha flag to the given value"""
    def locked(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief locked checks whether the Node is locked. A locked node cannot be changed.
        @return true if the Node is locked, false if it hasn't been locked."""
    def setLocked(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        set the Locked flag to the give value"""
    def hasExtents(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief does the node have any content in it?
        @return if node has any content in it"""
    def name(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the user-visible name of this node."""
    def setName(self, name: str) -> None:
        # type: (name) -> None:
        """@access public Q_SLOTS
        rename the Node to the given name"""
    def opacity(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        return the opacity of the Node. The opacity is a value between 0 and 255."""
    def setOpacity(self, value: int) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        set the opacity of the Node to the given value. The opacity is a value between 0 and 255."""
    def parentNode(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
        return the Node that is the parent of the current Node, or 0 if this is the root Node."""
    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return The type of the node. Valid types are: <ul>  <li>paintlayer  <li>grouplayer  <li>filelayer  <li>filterlayer  <li>filllayer  <li>clonelayer  <li>vectorlayer  <li>transparencymask  <li>filtermask  <li>transformmask  <li>selectionmask  <li>colorizemask </ul> If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def icon(self) -> QIcon:
        # type: () -> QIcon:
        """@access public Q_SLOTS
         @brief icon
        @return the icon associated with the layer."""
    def visible(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        Check whether the current Node is visible in the layer stack"""
    def hasKeyframeAtTime(self, frameNumber: int) -> bool:
        # type: (frameNumber) -> bool:
        """@access public Q_SLOTS
        Check to see if frame number on layer is a keyframe"""
    def setVisible(self, visible: bool) -> None:
        # type: (visible) -> None:
        """@access public Q_SLOTS
        Set the visibility of the current node to @param visible"""
    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        # type: (x, y, w, h) -> QByteArray:
        """@access public Q_SLOTS
         @brief pixelData reads the given rectangle from the Node's paintable pixels, if those exist, and returns it as a byte array. The pixel data starts top-left, and is ordered row-first. The byte array can be interpreted as follows: 8 bits images have one byte per channel, and as many bytes as there are channels. 16 bits integer images have two bytes per channel, representing an unsigned short. 16 bits float images have two bytes per channel, representing a half, or 16 bits float. 32 bits float images have four bytes per channel, representing a float. You can read outside the node boundaries; those pixels will be transparent black. The order of channels is: <ul> <li>Integer RGBA: Blue, Green, Red, Alpha <li>Float RGBA: Red, Green, Blue, Alpha <li>GrayA: Gray, Alpha <li>Selection: selectedness <li>LabA: L, a, b, Alpha <li>CMYKA: Cyan, Magenta, Yellow, Key, Alpha <li>XYZA: X, Y, Z, A <li>YCbCrA: Y, Cb, Cr, Alpha </ul> The byte array is a copy of the original node data. In Python, you can use bytes, bytearray and the struct module to interpret the data and construct, for instance, a Pillow Image object. If you read the pixeldata of a mask, a filter or generator layer, you get the selection bytes, which is one channel with values in the range from 0..255. If you want to change the pixels of a node you can write the pixels back after manipulation with setPixelData(). This will only succeed on nodes with writable pixel data, e.g not on groups or file layers.
        @param x x position from where to start reading
        @param y y position from where to start reading
        @param w row length to read
        @param h number of rows to read
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def pixelDataAtTime(self, x: int, y: int, w: int, h: int, time: int) -> QByteArray:
        # type: (x, y, w, h, time) -> QByteArray:
        """@access public Q_SLOTS
         @brief pixelDataAtTime a basic function to get pixeldata from an animated node at a given time.
        @param x the position from the left to start reading.
        @param y the position from the top to start reader
        @param w the row length to read
        @param h the number of rows to read
        @param time the frame number
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def projectionPixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        # type: (x, y, w, h) -> QByteArray:
        """@access public Q_SLOTS
         @brief projectionPixelData reads the given rectangle from the Node's projection (that is, what the node looks like after all sub-Nodes (like layers in a group or masks on a layer) have been applied, and returns it as a byte array. The pixel data starts top-left, and is ordered row-first. The byte array can be interpreted as follows: 8 bits images have one byte per channel, and as many bytes as there are channels. 16 bits integer images have two bytes per channel, representing an unsigned short. 16 bits float images have two bytes per channel, representing a half, or 16 bits float. 32 bits float images have four bytes per channel, representing a float. You can read outside the node boundaries; those pixels will be transparent black. The order of channels is: <ul> <li>Integer RGBA: Blue, Green, Red, Alpha <li>Float RGBA: Red, Green, Blue, Alpha <li>GrayA: Gray, Alpha <li>Selection: selectedness <li>LabA: L, a, b, Alpha <li>CMYKA: Cyan, Magenta, Yellow, Key, Alpha <li>XYZA: X, Y, Z, A <li>YCbCrA: Y, Cb, Cr, Alpha </ul> The byte array is a copy of the original node data. In Python, you can use bytes, bytearray and the struct module to interpret the data and construct, for instance, a Pillow Image object. If you read the projection of a mask, you get the selection bytes, which is one channel with values in the range from 0..255. If you want to change the pixels of a node you can write the pixels back after manipulation with setPixelData(). This will only succeed on nodes with writable pixel data, e.g not on groups or file layers.
        @param x x position from where to start reading
        @param y y position from where to start reading
        @param w row length to read
        @param h number of rows to read
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def setPixelData(
        self, value: Union[QByteArray, bytes, bytearray], x: int, y: int, w: int, h: int
    ) -> bool:
        # type: (value, x, y, w, h) -> bool:
        """@access public Q_SLOTS
         @brief setPixelData writes the given bytes, of which there must be enough, into the Node, if the Node has writable pixel data: <ul> <li>paint layer: the layer's original pixels are overwritten <li>filter layer, generator layer, any mask: the embedded selection's pixels are overwritten. <b>Note:</b> for these </ul> File layers, Group layers, Clone layers cannot be written to. Calling setPixelData on those layer types will silently do nothing.
        @param value the byte array representing the pixels. There must be enough bytes available. Krita will take the raw pointer from the QByteArray and start reading, not stopping before (number of channels * size of channel * w * h) bytes are read.
        @param x the x position to start writing from
        @param y the y position to start writing from
        @param w the width of each row
        @param h the number of rows to write
        @return true if writing the pixeldata worked"""
    def bounds(self) -> QRect:
        # type: () -> QRect:
        """@access public Q_SLOTS
         @brief bounds return the exact bounds of the node's paint device
        @return the bounds, or an empty QRect if the node has no paint device or is empty."""
    def move(self, x: int, y: int) -> None:
        # type: (x, y) -> None:
        """@access public Q_SLOTS
        move the pixels to the given x, y location in the image coordinate space."""
    def position(self) -> QPoint:
        # type: () -> QPoint:
        """@access public Q_SLOTS
         @brief position returns the position of the paint device of this node. The position is always 0,0 unless the layer has been moved. If you want to know the topleft position of the rectangle around the actual non-transparent pixels in the node, use bounds().
        @return the top-left position of the node"""
    def remove(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @brief remove removes this node from its parent image."""
    def duplicate(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
         @brief duplicate returns a full copy of the current node. The node is not inserted in the graphic
        @return a valid Node object or 0 if the node couldn't be duplicated."""
    def save(
        self,
        filename: str,
        xRes: float,
        yRes: float,
        exportConfiguration: "InfoObject",
        exportRect: QRect = QRect(),
    ) -> bool:
        # type: (filename, xRes, yRes, exportConfiguration, exportRect) -> bool:
        """@access public Q_SLOTS
         @brief save exports the given node with this filename. The extension of the filename determines the filetype.
        @param filename the filename including extension
        @param xRes the horizontal resolution in pixels per pt (there are 72 pts in an inch)
        @param yRes the horizontal resolution in pixels per pt (there are 72 pts in an inch)
        @param exportConfiguration a configuration object appropriate to the file format.
        @param exportRect the export bounds for saving a node as a QRect If \p exportRect is empty, then save exactBounds() of the node. If you'd like to save the image- aligned area of the node, just pass image->bounds() there. See Document->exportImage for InfoObject details.
        @return true if saving succeeded, false if it failed."""
    def mergeDown(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
        @brief mergeDown merges the given node with the first visible node underneath this node in the layerstack. This will drop all per-layer metadata.
        """
    def scaleNode(self, origin: QPointF, width: int, height: int, strategy: str) -> None:
        # type: (origin, width, height, strategy) -> None:
        """@access public Q_SLOTS
         @brief scaleNode
        @param origin the origin point
        @param width the width
        @param height the height
        @param strategy the scaling strategy. There's several ones amongst these that aren't available in the regular UI. <ul> <li>Hermite</li> <li>Bicubic - Adds pixels using the color of surrounding pixels. Produces smoother tonal gradations than Bilinear.</li> <li>Box - Replicate pixels in the image. Preserves all the original detail, but can produce jagged effects.</li> <li>Bilinear - Adds pixels averaging the color values of surrounding pixels. Produces medium quality results when the image is scaled from half to two times the original size.</li> <li>Bell</li> <li>BSpline</li> <li>Lanczos3 - Offers similar results than Bicubic, but maybe a little bit sharper. Can produce light and dark halos along strong edges.</li> <li>Mitchell</li> </ul>
        """
    def rotateNode(self, radians: float) -> None:
        # type: (radians) -> None:
        """@access public Q_SLOTS
         @brief rotateNode rotate this layer by the given radians.
        @param radians amount the layer should be rotated in, in radians."""
    def cropNode(self, x: int, y: int, w: int, h: int) -> None:
        # type: (x, y, w, h) -> None:
        """@access public Q_SLOTS
         @brief cropNode crop this layer.
        @param x the left edge of the cropping rectangle.
        @param y the top edge of the cropping rectangle
        @param w the right edge of the cropping rectangle
        @param h the bottom edge of the cropping rectangle"""
    def shearNode(self, angleX: float, angleY: float) -> None:
        # type: (angleX, angleY) -> None:
        """@access public Q_SLOTS
         @brief shearNode perform a shear operation on this node.
        @param angleX the X-angle in degrees to shear by
        @param angleY the Y-angle in degrees to shear by"""
    def thumbnail(self, w: int, h: int) -> QImage:
        # type: (w, h) -> QImage:
        """@access public Q_SLOTS
         @brief thumbnail create a thumbnail of the given dimensions. The thumbnail is sized according to the layer dimensions, not the image dimensions. If the requested size is too big a null QImage is created. If the current node cannot generate a thumbnail, a transparent QImage of the requested size is generated.
        @return a QImage representing the layer contents."""
    def layerStyleToAsl(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief layerStyleToAsl retreive the current layer's style in ASL format.
        @return a QString in ASL format representing the layer style."""
    def setLayerStyleFromAsl(self, asl: str) -> bool:
        # type: (asl) -> bool:
        """@access public Q_SLOTS
         @brief setLayerStyleFromAsl set a new layer style for this node.
        @param aslContent a string formatted in ASL format containing the layer style
        @return true if layer style was set, false if failed."""
    def index(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief index the index of the node inside the parent
        @return an integer representing the node's index inside the parent"""
    def uniqueId(self) -> QUuid:
        # type: () -> QUuid:
        """@access public Q_SLOTS
         @brief uniqueId uniqueId of the node
        @return a QUuid representing a unique id to identify the node"""
    def paintDevice(self) -> "KisPaintDeviceSP":
        # type: () -> 'KisPaintDeviceSP':
        """@access private
         @brief paintDevice gives access to the internal paint device of this Node
        @return the paintdevice or 0 if the node does not have an editable paint device."""

class ManagedColor(QObject):
    """* @brief The ManagedColor class is a class to handle colors that are color managed. A managed color is a color of which we know the model(RGB, LAB, CMYK, etc), the bitdepth and the specific properties of its colorspace, such as the whitepoint, chromacities, trc, etc, as represented by the color profile. Krita has two color management systems. LCMS and OCIO. LCMS is the one handling the ICC profile stuff, and the major one handling that ManagedColor deals with. OCIO support is only in the display of the colors. ManagedColor has some support for it in colorForCanvas() All colors in Krita are color managed. QColors are understood as RGB-type colors in the sRGB space. We recommend you make a color like this:
    @code
    colorYellow = ManagedColor("RGBA", "U8", "")
    QVector<float> yellowComponents = colorYellow.components()
    yellowComponents[0] = 1.0
    yellowComponents[1] = 1.0
    yellowComponents[2] = 0
    yellowComponents[3] = 1.0
    colorYellow.setComponents(yellowComponents)
    QColor yellow = colorYellow.colorForCanvas(canvas)
    @endcode"""

    def __init__(self) -> "KisPaintDeviceSP":
        """@brief paintDevice gives access to the internal paint device of this Node
        @return the paintdevice or 0 if the node does not have an editable paint device."""
    def colorForCanvas(self, canvas: "Canvas") -> QColor:
        # type: (canvas) -> QColor:
        """@access public
         @brief colorForCanvas
        @param canvas the canvas whose color management you'd like to use. In Krita, different views have separate canvasses, and these can have different OCIO configurations active.
        @return the QColor as it would be displaying on the canvas. This result can be used to draw widgets with the correct configuration applied.
        """
    @staticmethod
    def fromQColor(qcolor: QColor, canvas: "Canvas" = 0) -> "ManagedColor":
        # type: (qcolor, canvas) ->  ManagedColor:
        """@access public
         @brief fromQColor is the (approximate) reverse of colorForCanvas()
        @param qcolor the QColor to convert to a KoColor.
        @param canvas the canvas whose color management you'd like to use.
        @return the approximated ManagedColor, to use for canvas resources."""
    def colorDepth(self) -> str:
        # type: () -> str:
        """@access public
         colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @return the color depth."""
    def colorModel(self) -> str:
        # type: () -> str:
        """@access public
         @brief colorModel retrieve the current color model of this document: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @return the internal color model string."""
    def colorProfile(self) -> str:
        # type: () -> str:
        """@access public
        @return the name of the current color profile"""
    def setColorProfile(self, colorProfile: str) -> bool:
        # type: (colorProfile) -> bool:
        """@access public
         @brief setColorProfile set the color profile of the image to the given profile. The profile has to be registered with krita and be compatible with the current color model and depth; the image data is <i>not</i> converted.
        @param colorProfile
        @return false if the colorProfile name does not correspond to to a registered profile or if assigning the profile failed.
        """
    def setColorSpace(self, colorModel: str, colorDepth: str, colorProfile: str) -> bool:
        # type: (colorModel, colorDepth, colorProfile) -> bool:
        """@access public
         @brief setColorSpace convert the nodes and the image to the given colorspace. The conversion is done with Perceptual as intent, High Quality and No LCMS Optimizations as flags and no blackpoint compensation.
        @param colorModel A string describing the color model of the image: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @param colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @param colorProfile a valid color profile for this color model and color depth combination.
        @return false the combination of these arguments does not correspond to a colorspace."""
    def components(self) -> List[float]:
        # type: () -> List[float]:
        """@access public
         @brief components
        @return a QVector containing the channel/components of this color normalized. This includes the alphachannel.
        """
    def componentsOrdered(self) -> List[float]:
        # type: () -> List[float]:
        """@access public
         @brief componentsOrdered()
        @return same as Components, except the values are ordered to the display."""
    def setComponents(self, values: List[float]) -> None:
        # type: (values) -> None:
        """@access public
         @brief setComponents Set the channel/components with normalized values. For integer colorspace, this obviously means the limit is between 0.0-1.0, but for floating point colorspaces, 2.4 or 103.5 are still meaningful (if bright) values.
        @param values the QVector containing the new channel/component values. These should be normalized.
        """
    def toXML(self) -> str:
        # type: () -> str:
        """@access public
        Serialize this color following Create's swatch color specification available at https://web.archive.org/web/20110826002520/http://create.freedesktop.org/wiki/Swatches_-_color_file_format/Draft
        """
    def fromXML(self, xml: str) -> None:
        # type: (xml) -> None:
        """@access public
         Unserialize a color following Create's swatch color specification available at https://web.archive.org/web/20110826002520/http://create.freedesktop.org/wiki/Swatches_-_color_file_format/Draft
        @param xml an XML color
        @return the unserialized color, or an empty color object if the function failed         to unserialize the color
        """
    def toQString(self) -> str:
        # type: () -> str:
        """@access public
         @brief toQString create a user-visible string of the channel names and the channel values
        @return a string that can be used to display the values of this color to the user."""

class Krita(QObject):
    """* Krita is a singleton class that offers the root access to the Krita object hierarchy. The Krita.instance() is aliased as two builtins: Scripter and Application."""

    def activeDocument(self) -> "Document":
        # type: () -> Document:
        """@access public Q_SLOTS
        @return the currently active document, if there is one."""
    def setActiveDocument(self, value: "Document") -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setActiveDocument activates the first view that shows the given document
        @param value the document we want to activate"""
    def batchmode(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief batchmode determines whether the script is run in batch mode. If batchmode is true, scripts should now show messageboxes or dialog boxes. Note that this separate from Document.setBatchmode(), which determines whether export/save option dialogs are shown.
        @return true if the script is run in batchmode"""
    def setBatchmode(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setBatchmode sets the batchmode to @param value; if true, scripts should not show dialogs or messageboxes.
        """
    def actions(self) -> List[QAction]:
        # type: () -> List[QAction]:
        """@access public Q_SLOTS
        @return return a list of all actions for the currently active mainWindow."""
    def action(self, name: str) -> QAction:
        # type: (name) -> QAction:
        """@access public Q_SLOTS
        @return the action that has been registered under the given name, or 0 if no such action exists.
        """
    def documents(self) -> List["Document"]:
        # type: () -> List[Document]:
        """@access public Q_SLOTS
        @return a list of all open Documents"""
    def dockers(self) -> List[QDockWidget]:
        # type: () -> List[QDockWidget]:
        """@access public Q_SLOTS
        @return a list of all the dockers"""
    def filters(self) -> List[str]:
        # type: () -> List[str]:
        """@access public Q_SLOTS
         @brief Filters are identified by an internal name. This function returns a list of all existing registered filters.
        @return a list of all registered filters"""
    def filter(self, name: str) -> "Filter":
        # type: (name) -> Filter:
        """@access public Q_SLOTS
         @brief filter construct a Filter object with a default configuration.
        @param name the name of the filter. Use Krita.instance().filters() to get a list of all possible filters.
        @return the filter or None if there is no such filter."""
    def colorModels(self) -> List[str]:
        # type: () -> List[str]:
        """@access public Q_SLOTS
         @brief colorModels creates a list with all color models id's registered.
        @return a list of all color models or a empty list if there is no such color models."""
    def colorDepths(self, colorModel: str) -> List[str]:
        # type: (colorModel) -> List[str]:
        """@access public Q_SLOTS
         @brief colorDepths creates a list with the names of all color depths compatible with the given color model.
        @param colorModel the id of a color model.
        @return a list of all color depths or a empty list if there is no such color depths."""
    def filterStrategies(self) -> List[str]:
        # type: () -> List[str]:
        """@access public Q_SLOTS
         @brief filterStrategies Retrieves all installed filter strategies. A filter strategy is used when transforming (scaling, shearing, rotating) an image to calculate the value of the new pixels. You can use th
        @return the id's of all available filters."""
    def profiles(self, colorModel: str, colorDepth: str) -> List[str]:
        # type: (colorModel, colorDepth) -> List[str]:
        """@access public Q_SLOTS
         @brief profiles creates a list with the names of all color profiles compatible with the given color model and color depth.
        @param colorModel A string describing the color model of the image: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @param colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @return a list with valid names"""
    def addProfile(self, profilePath: str) -> bool:
        # type: (profilePath) -> bool:
        """@access public Q_SLOTS
         @brief addProfile load the given profile into the profile registry.
        @param profilePath the path to the profile.
        @return true if adding the profile succeeded."""
    def notifier(self) -> "Notifier":
        # type: () -> Notifier:
        """@access public Q_SLOTS
         @brief notifier the Notifier singleton emits signals when documents are opened and closed, the configuration changes, views are opened and closed or windows are opened.
        @return the notifier object"""
    def version(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief version Determine the version of Krita Usage: print(Application.version ())
        @return the version string including git sha1 if Krita was built from git"""
    def views(self) -> List["View"]:
        # type: () -> List[View]:
        """@access public Q_SLOTS
        @return a list of all views. A Document can be shown in more than one view."""
    def activeWindow(self) -> "Window":
        # type: () -> Window:
        """@access public Q_SLOTS
        @return the currently active window or None if there is no window"""
    def windows(self) -> List["Window"]:
        # type: () -> List[Window]:
        """@access public Q_SLOTS
        @return a list of all windows"""
    def resources(self, type: str) -> Dict[str, "Resource"]:
        # type: (type) -> Dict[str, Resource]:
        """@access public Q_SLOTS
         @brief resources returns a list of Resource objects of the given type
        @param type Valid types are: <ul> <li>pattern</li> <li>gradient</li> <li>brush</li> <li>preset</li> <li>palette</li> <li>workspace</li> </ul>
        """
    def recentDocuments(self) -> List[str]:
        # type: () -> List[str]:
        """@access public Q_SLOTS
        @brief return all recent documents registered in the RecentFiles group of the kritarc"""
    def createDocument(
        self,
        width: int,
        height: int,
        name: str,
        colorModel: str,
        colorDepth: str,
        profile: str,
        resolution: float,
    ) -> "Document":
        # type: (width, height, name, colorModel, colorDepth, profile, resolution) -> Document:
        """@access public Q_SLOTS
         @brief createDocument creates a new document and image and registers the document with the Krita application. Unless you explicitly call Document::close() the document will remain known to the Krita document registry. The document and its image will only be deleted when Krita exits. The document will have one transparent layer. To create a new document and show it, do something like:
        @code
        from Krita import *

        def add_document_to_window():
            d = Application.createDocument(100, 100, "Test", "RGBA", "U8", "", 120.0)
            Application.activeWindow().addView(d)

        add_document_to_window()
        @endcode
        @param width the width in pixels
        @param height the height in pixels
        @param name the name of the image (not the filename of the document)
        @param colorModel A string describing the color model of the image: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @param colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @param profile The name of an icc profile that is known to Krita. If an empty string is passed, the default is taken.
        @param resolution the resolution in points per inch.
        @return the created document."""
    def openDocument(self, filename: str) -> "Document":
        # type: (filename) -> Document:
        """@access public Q_SLOTS
         @brief openDocument creates a new Document, registers it with the Krita application and loads the given file.
        @param filename the file to open in the document
        @return the document"""
    def openWindow(self) -> "Window":
        # type: () -> Window:
        """@access public Q_SLOTS
        @brief openWindow create a new main window. The window is not shown by default."""
    def addExtension(self, extension: "Extension") -> None:
        # type: (extension) -> None:
        """@access public Q_SLOTS
         @brief addExtension add the given plugin to Krita. There will be a single instance of each Extension in the Krita process.
        @param extension the extension to add."""
    def extensions(self) -> List["Extension"]:
        # type: () -> List[Extension]:
        """@access public Q_SLOTS
        return a list with all registered extension objects."""
    def addDockWidgetFactory(self, factory: "DockWidgetFactoryBase") -> None:
        # type: (factory) -> None:
        """@access public Q_SLOTS
         @brief addDockWidgetFactory Add the given docker factory to the application. For scripts loaded on startup, this means that every window will have one of the dockers created by the factory.
        @param factory The factory object."""
    def writeSetting(self, group: str, name: str, value: str) -> None:
        # type: (group, name, value) -> None:
        """@access public Q_SLOTS
         @brief writeSetting write the given setting under the given name to the kritarc file in the given settings group.
        @param group The group the setting belongs to. If empty, then the setting is written in the general section
        @param name The name of the setting
        @param value The value of the setting. Script settings are always written as strings."""
    def readSetting(self, group: str, name: str, defaultValue: str) -> str:
        # type: (group, name, defaultValue) -> str:
        """@access public Q_SLOTS
         @brief readSetting read the given setting value from the kritarc file.
        @param group The group the setting is part of. If empty, then the setting is read from the general group.
        @param name The name of the setting
        @param defaultValue The default value of the setting
        @return a string representing the setting."""
    def icon(self, iconName: str) -> QIcon:
        # type: (iconName) -> QIcon:
        """@access public Q_SLOTS
         @brief icon This allows you to get icons from Krita's internal icons.
        @param iconName name of the icon.
        @return the icon related to this name."""
    @staticmethod
    def instance() -> "Krita":
        # type: () ->  Krita:
        """@access public Q_SLOTS
        @brief instance retrieve the singleton instance of the Application object."""
    def mainWindowIsBeingCreated(self, window: "KisMainWindow") -> None:
        # type: (window) -> None:
        """@access private Q_SLOTS
        This is called from the constructor of the window, before the xmlgui file is loaded"""

class InfoObject(QObject):
    """* InfoObject wrap a properties map. These maps can be used to set the configuration for filters."""

    def __init__(self, window: "KisMainWindow") -> None:
        """This is called from the constructor of the window, before the xmlgui file is loaded"""
    def properties(self) -> Dict[str, QVariant]:
        # type: () -> Dict[str, QVariant]:
        """@access public
        Return all properties this InfoObject manages."""
    def setProperties(self, propertyMap: Dict[str, QVariant]) -> None:
        # type: (propertyMap) -> None:
        """@access public
        Add all properties in the @p propertyMap to this InfoObject"""
    def setProperty(self, key: str, value: QVariant) -> None:
        # type: (key, value) -> None:
        """@access public Q_SLOTS
        set the property identified by @p key to @p value If you want create a property that represents a color, you can use a QColor or hex string, as defined in https://doc.qt.io/qt-5/qcolor.html#setNamedColor.
        """
    def property(self, key: str) -> QVariant:
        # type: (key) -> QVariant:
        """@access public Q_SLOTS
        return the value for the property identified by key, or None if there is no such key."""
    def configuration(self) -> "KisPropertiesConfigurationSP":
        # type: () -> 'KisPropertiesConfigurationSP':
        """@access private
         @brief configuration gives access to the internal configuration object. Must be used used internally in libkis
        @return the internal configuration object."""

class Filter(QObject):
    """* Filter: represents a filter and its configuration. A filter is identified by an internal name. The configuration for each filter is defined as an InfoObject: a map of name and value pairs. Currently available filters are: 'autocontrast', 'blur', 'bottom edge detections', 'brightnesscontrast', 'burn', 'colorbalance', 'colortoalpha', 'colortransfer', 'desaturate', 'dodge', 'emboss', 'emboss all directions', 'emboss horizontal and vertical', 'emboss horizontal only', 'emboss laplascian', 'emboss vertical only', 'gaussian blur', 'gaussiannoisereducer', 'gradientmap', 'halftone', 'hsvadjustment', 'indexcolors', 'invert', 'left edge detections', 'lens blur', 'levels', 'maximize', 'mean removal', 'minimize', 'motion blur', 'noise', 'normalize', 'oilpaint', 'perchannel', 'phongbumpmap', 'pixelize', 'posterize', 'raindrops', 'randompick', 'right edge detections', 'roundcorners', 'sharpen', 'smalltiles', 'sobel', 'threshold', 'top edge detections', 'unsharp', 'wave', 'waveletnoisereducer']"""

    def __init__(self) -> "KisPropertiesConfigurationSP":
        """@brief configuration gives access to the internal configuration object. Must be used used internally in libkis
        @return the internal configuration object."""
    def name(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief name the internal name of this filter.
        @return the name."""
    def setName(self, name: str) -> None:
        # type: (name) -> None:
        """@access public Q_SLOTS
        @brief setName set the filter's name to the given name."""
    def configuration(self) -> "InfoObject":
        # type: () -> InfoObject:
        """@access public Q_SLOTS
        @return the configuration object for the filter"""
    def setConfiguration(self, value: "InfoObject") -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setConfiguration set the configuration object for the filter"""
    def apply(self, node: "Node", x: int, y: int, w: int, h: int) -> bool:
        # type: (node, x, y, w, h) -> bool:
        """@access public Q_SLOTS
         @brief Apply the filter to the given node.
        @param node the node to apply the filter to
        @param x
        @param y
        @param w
        @param h describe the rectangle the filter should be apply. This is always in image pixel coordinates and not relative to the x, y of the node.
        @return @c true if the filter was applied successfully, or
        @c false if the filter could not be applied because the node is locked or does not have an editable paint device.
        """
    def startFilter(self, node: "Node", x: int, y: int, w: int, h: int) -> bool:
        # type: (node, x, y, w, h) -> bool:
        """@access public Q_SLOTS
         @brief startFilter starts the given filter on the given node.
        @param node the node to apply the filter to
        @param x
        @param y
        @param w
        @param h describe the rectangle the filter should be apply. This is always in image pixel coordinates and not relative to the x, y of the node.
        """

class Extension(QObject):
    """* An Extension is the base for classes that extend Krita. An Extension is loaded on startup, when the setup() method will be executed. The extension instance should be added to the Krita Application object using Krita.instance().addViewExtension or Application.addViewExtension or Scripter.addViewExtension. Example:
    @code
    import sys
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from krita import *
    class HelloExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
    def hello(self):
        QMessageBox.information(QWidget(), "Test", "Hello! This is Krita " + Application.version())
    def setup(self):
        qDebug("Hello Setup")
    def createActions(self, window)
        action = window.createAction("hello")
        action.triggered.connect(self.hello)
    Scripter.addExtension(HelloExtension(Krita.instance()))
    @endcode"""

class Document(QObject):
    """* The Document class encapsulates a Krita Document/Image. A Krita document is an Image with a filename. Libkis does not differentiate between a document and an image, like Krita does internally."""

    def __init__(self, node: "Node", x: int, y: int, w: int, h: int) -> bool:
        """@brief startFilter starts the given filter on the given node.
        @param node the node to apply the filter to
        @param x
        @param y
        @param w
        @param h describe the rectangle the filter should be apply. This is always in image pixel coordinates and not relative to the x, y of the node.
        """
    def horizontalGuides(self) -> List[float]:
        # type: () -> List[float]:
        """@access public
         @brief horizontalGuides The horizontal guides.
        @return a list of the horizontal positions of guides."""
    def verticalGuides(self) -> List[float]:
        # type: () -> List[float]:
        """@access public
         @brief verticalGuides The vertical guide lines.
        @return a list of vertical guides."""
    def guidesVisible(self) -> bool:
        # type: () -> bool:
        """@access public
         @brief guidesVisible Returns guide visibility.
        @return whether the guides are visible."""
    def guidesLocked(self) -> bool:
        # type: () -> bool:
        """@access public
         @brief guidesLocked Returns guide lockedness.
        @return whether the guides are locked."""
    def clone(self) -> "Document":
        # type: () -> Document:
        """@access public Q_SLOTS
         @brief clone create a shallow clone of this document.
        @return a new Document that should be identical to this one in every respect."""
    def batchmode(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         Batchmode means that no actions on the document should show dialogs or popups.
        @return true if the document is in batchmode."""
    def setBatchmode(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        Set batchmode to @p value. If batchmode is true, then there should be no popups or dialogs shown to the user.
        """
    def activeNode(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
         @brief activeNode retrieve the node that is currently active in the currently active window
        @return the active node. If there is no active window, the first child node is returned."""
    def setActiveNode(self, value: "Node") -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setActiveNode make the given node active in the currently active view and window
        @param value the node to make active."""
    def topLevelNodes(self) -> List["Node"]:
        # type: () -> List[Node]:
        """@access public Q_SLOTS
        @brief toplevelNodes return a list with all top level nodes in the image graph"""
    def nodeByName(self, name: str) -> "Node":
        # type: (name) -> Node:
        """@access public Q_SLOTS
         @brief nodeByName searches the node tree for a node with the given name and returns it
        @param name the name of the node
        @return the first node with the given name or 0 if no node is found"""
    def nodeByUniqueID(self, id: QUuid) -> "Node":
        # type: (id) -> Node:
        """@access public Q_SLOTS
         @brief nodeByUniqueID searches the node tree for a node with the given name and returns it.
        @param uuid the unique id of the node
        @return the node with the given unique id, or 0 if no node is found."""
    def colorDepth(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @return the color depth."""
    def colorModel(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief colorModel retrieve the current color model of this document: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @return the internal color model string."""
    def colorProfile(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the name of the current color profile"""
    def setColorProfile(self, colorProfile: str) -> bool:
        # type: (colorProfile) -> bool:
        """@access public Q_SLOTS
         @brief setColorProfile set the color profile of the image to the given profile. The profile has to be registered with krita and be compatible with the current color model and depth; the image data is <i>not</i> converted.
        @param colorProfile
        @return false if the colorProfile name does not correspond to to a registered profile or if assigning the profile failed.
        """
    def setColorSpace(self, colorModel: str, colorDepth: str, colorProfile: str) -> bool:
        # type: (colorModel, colorDepth, colorProfile) -> bool:
        """@access public Q_SLOTS
         @brief setColorSpace convert the nodes and the image to the given colorspace. The conversion is done with Perceptual as intent, High Quality and No LCMS Optimizations as flags and no blackpoint compensation.
        @param colorModel A string describing the color model of the image: <ul> <li>A: Alpha mask</li> <li>RGBA: RGB with alpha channel (The actual order of channels is most often BGR!)</li> <li>XYZA: XYZ with alpha channel</li> <li>LABA: LAB with alpha channel</li> <li>CMYKA: CMYK with alpha channel</li> <li>GRAYA: Gray with alpha channel</li> <li>YCbCrA: YCbCr with alpha channel</li> </ul>
        @param colorDepth A string describing the color depth of the image: <ul> <li>U8: unsigned 8 bits integer, the most common type</li> <li>U16: unsigned 16 bits integer</li> <li>F16: half, 16 bits floating point. Only available if Krita was built with OpenEXR</li> <li>F32: 32 bits floating point</li> </ul>
        @param colorProfile a valid color profile for this color model and color depth combination.
        @return false the combination of these arguments does not correspond to a colorspace."""
    def backgroundColor(self) -> QColor:
        # type: () -> QColor:
        """@access public Q_SLOTS
         @brief backgroundColor returns the current background color of the document. The color will also include the opacity.
        @return QColor"""
    def setBackgroundColor(self, color: QColor) -> bool:
        # type: (color) -> bool:
        """@access public Q_SLOTS
         @brief setBackgroundColor sets the background color of the document. It will trigger a projection update.
        @param color A QColor. The color will be converted from sRGB.
        @return bool"""
    def documentInfo(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief documentInfo creates and XML document representing document and author information.
        @return a string containing a valid XML document with the right information about the document and author. The DTD can be found here: https://phabricator.kde.org/source/krita/browse/master/krita/dtd/
        @code
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE document-info PUBLIC '-//KDE//DTD document-info 1.1//EN' 'http://www.calligra.org/DTD/document-info-1.1.dtd'>
        <document-info xmlns="http://www.calligra.org/DTD/document-info">
        <about>
         <title>My Document</title>
          <description></description>
          <subject></subject>
          <abstract><![CDATA[]]></abstract>
          <keyword></keyword>
          <initial-creator>Unknown</initial-creator>
          <editing-cycles>1</editing-cycles>
          <editing-time>35</editing-time>
          <date>2017-02-27T20:15:09</date>
          <creation-date>2017-02-27T20:14:33</creation-date>
          <language></language>
         </about>
         <author>
          <full-name>Boudewijn Rempt</full-name>
          <initial></initial>
          <author-title></author-title>
          <email></email>
          <telephone></telephone>
          <telephone-work></telephone-work>
          <fax></fax>
          <country></country>
          <postal-code></postal-code>
          <city></city>
          <street></street>
          <position></position>
          <company></company>
         </author>
        </document-info>
        @endcode"""
    def setDocumentInfo(self, document: str) -> None:
        # type: (document) -> None:
        """@access public Q_SLOTS
         @brief setDocumentInfo set the Document information to the information contained in document
        @param document A string containing a valid XML document that conforms to the document-info DTD that can be found here: https://phabricator.kde.org/source/krita/browse/master/krita/dtd/
        """
    def fileName(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the full path to the document, if it has been set."""
    def setFileName(self, value: str) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setFileName set the full path of the document to @param value"""
    def height(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the height of the image in pixels"""
    def setHeight(self, value: int) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setHeight resize the document to @param value height. This is a canvas resize, not a scale.
        """
    def name(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
        @return the name of the document. This is the title field in the @ref documentInfo"""
    def setName(self, value: str) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setName sets the name of the document to @p value. This is the title field in the @ref documentInfo
        """
    def resolution(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the resolution in pixels per inch"""
    def setResolution(self, value: int) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setResolution set the resolution of the image; this does not scale the image
        @param value the resolution in pixels per inch"""
    def rootNode(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
         @brief rootNode the root node is the invisible group layer that contains the entire node hierarchy.
        @return the root of the image"""
    def selection(self) -> "Selection":
        # type: () -> Selection:
        """@access public Q_SLOTS
         @brief selection Create a Selection object around the global selection, if there is one.
        @return the global selection or None if there is no global selection."""
    def setSelection(self, value: "Selection") -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setSelection set or replace the global selection
        @param value a valid selection object."""
    def width(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the width of the image in pixels."""
    def setWidth(self, value: int) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setWidth resize the document to @param value width. This is a canvas resize, not a scale.
        """
    def xOffset(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the left edge of the canvas in pixels."""
    def setXOffset(self, x: int) -> None:
        # type: (x) -> None:
        """@access public Q_SLOTS
        @brief setXOffset sets the left edge of the canvas to @p x."""
    def yOffset(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
        @return the top edge of the canvas in pixels."""
    def setYOffset(self, y: int) -> None:
        # type: (y) -> None:
        """@access public Q_SLOTS
        @brief setYOffset sets the top edge of the canvas to @p y."""
    def xRes(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
        @return xRes the horizontal resolution of the image in pixels per inch"""
    def setXRes(self, xRes: float) -> None:
        # type: (xRes) -> None:
        """@access public Q_SLOTS
        @brief setXRes set the horizontal resolution of the image to xRes in pixels per inch"""
    def yRes(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
        @return yRes the vertical resolution of the image in pixels per inch"""
    def setYRes(self, yRes: float) -> None:
        # type: (yRes) -> None:
        """@access public Q_SLOTS
        @brief setYRes set the vertical resolution of the image to yRes in pixels per inch"""
    def pixelData(self, x: int, y: int, w: int, h: int) -> QByteArray:
        # type: (x, y, w, h) -> QByteArray:
        """@access public Q_SLOTS
         @brief pixelData reads the given rectangle from the image projection and returns it as a byte array. The pixel data starts top-left, and is ordered row-first. The byte array can be interpreted as follows: 8 bits images have one byte per channel, and as many bytes as there are channels. 16 bits integer images have two bytes per channel, representing an unsigned short. 16 bits float images have two bytes per channel, representing a half, or 16 bits float. 32 bits float images have four bytes per channel, representing a float. You can read outside the image boundaries; those pixels will be transparent black. The order of channels is: <ul> <li>Integer RGBA: Blue, Green, Red, Alpha <li>Float RGBA: Red, Green, Blue, Alpha <li>LabA: L, a, b, Alpha <li>CMYKA: Cyan, Magenta, Yellow, Key, Alpha <li>XYZA: X, Y, Z, A <li>YCbCrA: Y, Cb, Cr, Alpha </ul> The byte array is a copy of the original image data. In Python, you can use bytes, bytearray and the struct module to interpret the data and construct, for instance, a Pillow Image object.
        @param x x position from where to start reading
        @param y y position from where to start reading
        @param w row length to read
        @param h number of rows to read
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def close(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief close Close the document: remove it from Krita's internal list of documents and close all views. If the document is modified, you should save it first. There will be no prompt for saving. After closing the document it becomes invalid.
        @return true if the document is closed."""
    def crop(self, x: int, y: int, w: int, h: int) -> None:
        # type: (x, y, w, h) -> None:
        """@access public Q_SLOTS
         @brief crop the image to rectangle described by @p x, @p y,
        @p w and @p h
        @param x x coordinate of the top left corner
        @param y y coordinate of the top left corner
        @param w width
        @param h height"""
    def exportImage(self, filename: str, exportConfiguration: "InfoObject") -> bool:
        # type: (filename, exportConfiguration) -> bool:
        """@access public Q_SLOTS
         @brief exportImage export the image, without changing its URL to the given path.
        @param filename the full path to which the image is to be saved
        @param exportConfiguration a configuration object appropriate to the file format. An InfoObject will used to that configuration. The supported formats have specific configurations that must be used when in batchmode. They are described below:\b png <ul> <li>alpha: bool (True or False) <li>compression: int (1 to 9) <li>forceSRGB: bool (True or False) <li>indexed: bool (True or False) <li>interlaced: bool (True or False) <li>saveSRGBProfile: bool (True or False) <li>transparencyFillcolor: rgb (Ex:[255,255,255]) </ul>\b jpeg <ul> <li>baseline: bool (True or False) <li>exif: bool (True or False) <li>filters: bool (['ToolInfo', 'Anonymizer']) <li>forceSRGB: bool (True or False) <li>iptc: bool (True or False) <li>is_sRGB: bool (True or False) <li>optimize: bool (True or False) <li>progressive: bool (True or False) <li>quality: int (0 to 100) <li>saveProfile: bool (True or False) <li>smoothing: int (0 to 100) <li>subsampling: int (0 to 3) <li>transparencyFillcolor: rgb (Ex:[255,255,255]) <li>xmp: bool (True or False) </ul>
        @return true if the export succeeded, false if it failed."""
    def flatten(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief flatten all layers in the image"""
    def resizeImage(self, x: int, y: int, w: int, h: int) -> None:
        # type: (x, y, w, h) -> None:
        """@access public Q_SLOTS
         @brief resizeImage resizes the canvas to the given left edge, top edge, width and height. Note: This doesn't scale, use scale image for that.
        @param x the new left edge
        @param y the new top edge
        @param w the new width
        @param h the new height"""
    def scaleImage(self, w: int, h: int, xres: int, yres: int, strategy: str) -> None:
        # type: (w, h, xres, yres, strategy) -> None:
        """@access public Q_SLOTS
         @brief scaleImage
        @param w the new width
        @param h the new height
        @param xres the new xres
        @param yres the new yres
        @param strategy the scaling strategy. There's several ones amongst these that aren't available in the regular UI. The list of filters is extensible and can be retrieved with Krita::filter <ul> <li>Hermite</li> <li>Bicubic - Adds pixels using the color of surrounding pixels. Produces smoother tonal gradations than Bilinear.</li> <li>Box - Replicate pixels in the image. Preserves all the original detail, but can produce jagged effects.</li> <li>Bilinear - Adds pixels averaging the color values of surrounding pixels. Produces medium quality results when the image is scaled from half to two times the original size.</li> <li>Bell</li> <li>BSpline</li> <li>Kanczos3 - Offers similar results than Bicubic, but maybe a little bit sharper. Can produce light and dark halos along strong edges.</li> <li>Mitchell</li> </ul>
        """
    def rotateImage(self, radians: float) -> None:
        # type: (radians) -> None:
        """@access public Q_SLOTS
         @brief rotateImage Rotate the image by the given radians.
        @param radians the amount you wish to rotate the image in radians"""
    def shearImage(self, angleX: float, angleY: float) -> None:
        # type: (angleX, angleY) -> None:
        """@access public Q_SLOTS
         @brief shearImage shear the whole image.
        @param angleX the X-angle in degrees to shear by
        @param angleY the Y-angle in degrees to shear by"""
    def save(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief save the image to its currently set path. The modified flag of the document will be reset
        @return true if saving succeeded, false otherwise."""
    def saveAs(self, filename: str) -> bool:
        # type: (filename) -> bool:
        """@access public Q_SLOTS
         @brief saveAs save the document under the @p filename. The document's filename will be reset to @p filename.
        @param filename the new filename (full path) for the document
        @return true if saving succeeded, false otherwise."""
    def createNode(self, name: str, nodeType: str) -> "Node":
        # type: (name, nodeType) -> Node:
        """@access public Q_SLOTS
         @brief createNode create a new node of the given type. The node is not added to the node hierarchy; you need to do that by finding the right parent node, getting its list of child nodes and adding the node in the right place, then calling Node::SetChildNodes
        @param name The name of the node
        @param nodeType The type of the node. Valid types are: <ul>  <li>paintlayer  <li>grouplayer  <li>filelayer  <li>filterlayer  <li>filllayer  <li>clonelayer  <li>vectorlayer  <li>transparencymask  <li>filtermask  <li>transformmask  <li>selectionmask </ul> When relevant, the new Node will have the colorspace of the image by default; that can be changed with Node::setColorSpace. The settings and selections for relevant layer and mask types can also be set after the Node has been created.
        @code
        d = Application.createDocument(1000, 1000, "Test", "RGBA", "U8", "", 120.0)
        root = d.rootNode();
        print(root.childNodes())
        l2 = d.createNode("layer2", "paintLayer")
        print(l2)
        root.addChildNode(l2, None)
        print(root.childNodes())
        @endcode
        @return the new Node."""
    def createGroupLayer(self, name: str) -> "GroupLayer":
        # type: (name) -> GroupLayer:
        """@access public Q_SLOTS
         @brief createGroupLayer Returns a grouplayer object. Grouplayers are nodes that can have other layers as children and have the passthrough mode.
        @param name the name of the layer.
        @return a GroupLayer object."""
    def createFileLayer(self, name: str, fileName: str, scalingMethod: str) -> "FileLayer":
        # type: (name, fileName, scalingMethod) -> FileLayer:
        """@access public Q_SLOTS
         @brief createFileLayer returns a layer that shows an external image.
        @param name name of the file layer.
        @param fileName the absolute filename of the file referenced. Symlinks will be resolved.
        @param scalingMethod how the dimensions of the file are interpreted        can be either "None", "ImageToSize" or "ImageToPPI"
        @return a FileLayer"""
    def createFilterLayer(
        self, name: str, filter: "Filter", selection: "Selection"
    ) -> "FilterLayer":
        # type: (name, filter, selection) -> FilterLayer:
        """@access public Q_SLOTS
         @brief createFilterLayer creates a filter layer, which is a layer that represents a filter applied non-destructively.
        @param name name of the filterLayer
        @param filter the filter that this filter layer will us.
        @param selection the selection.
        @return a filter layer object."""
    def createFillLayer(
        self, name: str, generatorName: str, configuration: "InfoObject", selection: "Selection"
    ) -> "FillLayer":
        # type: (name, generatorName, configuration, selection) -> FillLayer:
        """@access public Q_SLOTS
         @brief createFillLayer creates a fill layer object, which is a layer
        @param name
        @param generatorName - name of the generation filter.
        @param configuration - the configuration for the generation filter.
        @param selection - the selection.
        @return a filllayer object.
        @code
        from krita import *
        d = Krita.instance().activeDocument()
        i = InfoObject();
        i.setProperty("pattern", "Cross01.pat")
        s = Selection();
        s.select(0, 0, d.width(), d.height(), 255)
        n = d.createFillLayer("test", "pattern", i, s)
        r = d.rootNode();
        c = r.childNodes();
        r.addChildNode(n, c[0])
        d.refreshProjection()
        @endcode"""
    def createCloneLayer(self, name: str, source: "Node") -> "CloneLayer":
        # type: (name, source) -> CloneLayer:
        """@access public Q_SLOTS
         @brief createCloneLayer
        @param name
        @param source
        @return"""
    def createVectorLayer(self, name: str) -> "VectorLayer":
        # type: (name) -> VectorLayer:
        """@access public Q_SLOTS
         @brief createVectorLayer Creates a vector layer that can contain vector shapes.
        @param name the name of this layer.
        @return a VectorLayer."""
    def createFilterMask(
        self, name: str, filter: "Filter", selection_source: "Node"
    ) -> "FilterMask":
        # type: (name, filter, selection_source) -> FilterMask:
        """@access public Q_SLOTS
         @brief createFilterMask Creates a filter mask object that much like a filterlayer can apply a filter non-destructively.
        @param name the name of the layer.
        @param filter the filter assigned.
        @param selection_source a node from which the selection should be initialized
        @return a FilterMask"""
    def createSelectionMask(self, name: str) -> "SelectionMask":
        # type: (name) -> SelectionMask:
        """@access public Q_SLOTS
         @brief createSelectionMask Creates a selection mask, which can be used to store selections.
        @param name - the name of the layer.
        @return a SelectionMask"""
    def createTransparencyMask(self, name: str) -> "TransparencyMask":
        # type: (name) -> TransparencyMask:
        """@access public Q_SLOTS
         @brief createTransparencyMask Creates a transparency mask, which can be used to assign transparency to regions.
        @param name - the name of the layer.
        @return a TransparencyMask"""
    def createTransformMask(self, name: str) -> "TransformMask":
        # type: (name) -> TransformMask:
        """@access public Q_SLOTS
         @brief createTransformMask Creates a transform mask, which can be used to apply a transformation non-destructively.
        @param name - the name of the layer mask.
        @return a TransformMask"""
    def createColorizeMask(self, name: str) -> "ColorizeMask":
        # type: (name) -> ColorizeMask:
        """@access public Q_SLOTS
         @brief createColorizeMask Creates a colorize mask, which can be used to color fill via keystrokes.
        @param name - the name of the layer.
        @return a TransparencyMask"""
    def projection(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> QImage:
        # type: (x, y, w, h) -> QImage:
        """@access public Q_SLOTS
        @brief projection creates a QImage from the rendered image or a cutout rectangle."""
    def thumbnail(self, w: int, h: int) -> QImage:
        # type: (w, h) -> QImage:
        """@access public Q_SLOTS
         @brief thumbnail create a thumbnail of the given dimensions. If the requested size is too big a null QImage is created.
        @return a QImage representing the layer contents."""
    def lock(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        [low-level] Lock the image without waiting for all the internal job queues are processed WARNING: Don't use it unless you really know what you are doing! Use barrierLock() instead! Waits for all the **currently running** internal jobs to complete and locks the image for writing. Please note that this function does **not** wait for all the internal queues to process, so there might be some non-finished actions pending. It means that you just postpone these actions until you unlock() the image back. Until then, then image might easily be frozen in some inconsistent state. The only sane usage for this function is to lock the image for **emergency** processing, when some internal action or scheduler got hung up, and you just want to fetch some data from the image without races. In all other cases, please use barrierLock() instead!
        """
    def unlock(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Unlocks the image and starts/resumes all the pending internal jobs. If the image has been locked for a non-readOnly access, then all the internal caches of the image (e.g. lod-planes) are reset and regeneration jobs are scheduled.
        """
    def waitForDone(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Wait for all the internal image jobs to complete and return without locking the image. This function is handly for tests or other synchronous actions, when one needs to wait for the result of his actions.
        """
    def tryBarrierLock(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief Tries to lock the image without waiting for the jobs to finish Same as barrierLock(), but doesn't block execution of the calling thread until all the background jobs are finished. Instead, in case of presence of unfinished jobs in the queue, it just returns false
        @return whether the lock has been acquired"""
    def refreshProjection(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        Starts a synchronous recomposition of the projection: everything will wait until the image is fully recomputed.
        """
    def setHorizontalGuides(self, lines: List[float]) -> None:
        # type: (lines) -> None:
        """@access public Q_SLOTS
         @brief setHorizontalGuides replace all existing horizontal guides with the entries in the list.
        @param lines a list of floats containing the new guides."""
    def setVerticalGuides(self, lines: List[float]) -> None:
        # type: (lines) -> None:
        """@access public Q_SLOTS
         @brief setVerticalGuides replace all existing horizontal guides with the entries in the list.
        @param lines a list of floats containing the new guides."""
    def setGuidesVisible(self, visible: bool) -> None:
        # type: (visible) -> None:
        """@access public Q_SLOTS
         @brief setGuidesVisible set guides visible on this document.
        @param visible whether or not the guides are visible."""
    def setGuidesLocked(self, locked: bool) -> None:
        # type: (locked) -> None:
        """@access public Q_SLOTS
         @brief setGuidesLocked set guides locked on this document
        @param locked whether or not to lock the guides on this document."""
    def modified(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @brief modified returns true if the document has unsaved modifications."""
    def bounds(self) -> QRect:
        # type: () -> QRect:
        """@access public Q_SLOTS
         @brief bounds return the bounds of the image
        @return the bounds"""
    def importAnimation(self, files: List[str], firstFrame: int, step: int) -> bool:
        # type: (files, firstFrame, step) -> bool:
        """@access public Q_SLOTS
         Animation Related API****/


            /**
        @brief Import an image sequence of files from a directory. This will grab all images from the directory and import them with a potential offset (firstFrame) and step (images on 2s, 3s, etc)
        @returns whether the animation import was successful"""
    def framesPerSecond(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief frames per second of document
        @return the fps of the document"""
    def setFramesPerSecond(self, fps: int) -> None:
        # type: (fps) -> None:
        """@access public Q_SLOTS
        @brief set frames per second of document"""
    def setFullClipRangeStartTime(self, startTime: int) -> None:
        # type: (startTime) -> None:
        """@access public Q_SLOTS
        @brief set start time of animation"""
    def fullClipRangeStartTime(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get the full clip range start time
        @return full clip range start time"""
    def setFullClipRangeEndTime(self, endTime: int) -> None:
        # type: (endTime) -> None:
        """@access public Q_SLOTS
        @brief set full clip range end time"""
    def fullClipRangeEndTime(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get the full clip range end time
        @return full clip range end time"""
    def animationLength(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get total frame range for animation
        @return total frame range for animation"""
    def setPlayBackRange(self, start: int, stop: int) -> None:
        # type: (start, stop) -> None:
        """@access public Q_SLOTS
        @brief set temporary playback range of document"""
    def playBackStartTime(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get start time of current playback
        @return start time of current playback"""
    def playBackEndTime(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get end time of current playback
        @return end time of current playback"""
    def currentTime(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief get current frame selected of animation
        @return current frame selected of animation"""
    def setCurrentTime(self, time: int) -> None:
        # type: (time) -> None:
        """@access public Q_SLOTS
        @brief set current time of document's animation"""
    def annotationTypes(self) -> List[str]:
        # type: () -> List[str]:
        """@access public Q_SLOTS
        @brief annotationTypes returns the list of annotations present in the document. Each annotation type is unique.
        """
    def annotationDescription(self, type: str) -> str:
        # type: (type) -> str:
        """@access public Q_SLOTS
         @brief annotationDescription gets the pretty description for the current annotation
        @param type the type of the annotation
        @return a string that can be presented to the user"""
    def annotation(self, type: str) -> QByteArray:
        # type: (type) -> QByteArray:
        """@access public Q_SLOTS
         @brief annotation the actual data for the annotation for this type. It's a simple QByteArray, what's in it depends on the type of the annotation
        @param type the type of the annotation
        @return a bytearray, possibly empty if this type of annotation doesn't exist"""
    def setAnnotation(
        self, type: str, description: str, annotation: Union[QByteArray, bytes, bytearray]
    ) -> None:
        # type: (type, description, annotation) -> None:
        """@access public Q_SLOTS
         @brief setAnnotation Add the given annotation to the document
        @param type the unique type of the annotation
        @param description the user-visible description of the annotation
        @param annotation the annotation itself"""
    def removeAnnotation(self, type: str) -> None:
        # type: (type) -> None:
        """@access public Q_SLOTS
         @brief removeAnnotation remove the specified annotation from the image
        @param type the type defining the annotation"""

class Channel(QObject):
    """* A Channel represents a single channel in a Node. Krita does not use channels to store local selections: these are strictly the color and alpha channels."""

    def visible(self) -> bool:
        # type: () -> bool:
        """@access public
         @brief visible checks whether this channel is visible in the node
        @return the status of this channel"""
    def setVisible(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public
        @brief setvisible set the visibility of the channel to the given value."""
    def name(self) -> str:
        # type: () -> str:
        """@access public
        @return the name of the channel"""
    def position(self) -> int:
        # type: () -> int:
        """@access public
        @returns the position of the first byte of the channel in the pixel"""
    def channelSize(self) -> int:
        # type: () -> int:
        """@access public
        @return the number of bytes this channel takes"""
    def bounds(self) -> QRect:
        # type: () -> QRect:
        """@access public
        @return the exact bounds of the channel. This can be smaller than the bounds of the Node this channel is part of.
        """
    def pixelData(self, rect: QRect) -> QByteArray:
        # type: (rect) -> QByteArray:
        """@access public
        Read the values of the channel into the a byte array for each pixel in the rect from the Node this channel is part of, and returns it. Note that if Krita is built with OpenEXR and the Node has the 16 bits floating point channel depth type, Krita returns 32 bits float for every channel; the libkis scripting API does not support half.
        """
    def setPixelData(self, value: Union[QByteArray, bytes, bytearray], rect: QRect) -> None:
        # type: (value, rect) -> None:
        """@access public
         @brief setPixelData writes the given data to the relevant channel in the Node. This is only possible for Nodes that have a paintDevice, so nothing will happen when trying to write to e.g. a group layer. Note that if Krita is built with OpenEXR and the Node has the 16 bits floating point channel depth type, Krita expects to be given a 4 byte, 32 bits float for every channel; the libkis scripting API does not support half.
        @param value a byte array with exactly enough bytes.
        @param rect the rectangle to write the bytes into"""

class Canvas(QObject):
    """* Canvas wraps the canvas inside a view on an image/document. It is responsible for the view parameters of the document: zoom, rotation, mirror, wraparound and instant preview."""

    def zoomLevel(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
        @return the current zoomlevel. 1.0 is 100%."""
    def setZoomLevel(self, value: float) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setZoomLevel set the zoomlevel to the given @p value. 1.0 is 100%."""
    def resetZoom(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief resetZoom set the zoomlevel to 100%"""
    def rotation(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
        @return the rotation of the canvas in degrees."""
    def setRotation(self, angle: float) -> None:
        # type: (angle) -> None:
        """@access public Q_SLOTS
        @brief setRotation set the rotation of the canvas to the given  @param angle in degrees."""
    def resetRotation(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief resetRotation reset the canvas rotation."""
    def mirror(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @return return true if the canvas is mirrored, false otherwise."""
    def setMirror(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
        @brief setMirror turn the canvas mirroring on or off depending on @param value"""
    def wrapAroundMode(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @return true if the canvas is in wraparound mode, false if not. Only when OpenGL is enabled, is wraparound mode available.
        """
    def setWrapAroundMode(self, enable: bool) -> None:
        # type: (enable) -> None:
        """@access public Q_SLOTS
        @brief setWrapAroundMode set wraparound mode to  @param enable"""
    def levelOfDetailMode(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
        @return true if the canvas is in Instant Preview mode, false if not. Only when OpenGL is enabled, is Instant Preview mode available.
        """
    def setLevelOfDetailMode(self, enable: bool) -> None:
        # type: (enable) -> None:
        """@access public Q_SLOTS
        @brief setLevelOfDetailMode sets Instant Preview to @param enable"""
    def view(self) -> "View":
        # type: () -> View:
        """@access public Q_SLOTS
        @return the view that holds this canvas"""

class VectorLayer(Node):
    """* @brief The VectorLayer class A vector layer is a special layer that stores and shows vector shapes. Vector shapes all have their coordinates in points, which is a unit that represents 1/72th of an inch. Keep this in mind wen parsing the bounding box and position data."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return vectorlayer"""
    def shapes(self) -> List["Shape"]:
        # type: () -> List[Shape]:
        """@access public Q_SLOTS
         @brief shapes
        @return the list of top-level shapes in this vector layer."""
    def toSvg(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief toSvg convert the shapes in the layer to svg.
        @return the svg in a string."""
    def addShapesFromSvg(self, svg: str) -> List["Shape"]:
        # type: (svg) -> List[Shape]:
        """@access public Q_SLOTS
         @brief addShapesFromSvg add shapes to the layer from a valid svg.
        @param svg valid svg string.
        @return the list of shapes added to the layer from the svg."""
    def shapeAtPosition(self, position: QPointF) -> "Shape":
        # type: (position) -> Shape:
        """@access public Q_SLOTS
         @brief shapeAtPoint check if the position is located within any non-group shape's boundingBox() on the current layer.
        @param position a QPointF of the position.
        @return the shape at the position, or None if no shape is found."""
    def shapesInRect(
        self, rect: QRectF, omitHiddenShapes: bool = True, containedMode: bool = False
    ) -> List["Shape"]:
        # type: (rect, omitHiddenShapes, containedMode) -> List[Shape]:
        """@access public Q_SLOTS
         @brief shapeInRect get all non-group shapes that the shape's boundingBox() intersects or is contained within a given rectangle on the current layer.
        @param rect a QRectF
        @param omitHiddenShapes true if non-visible() shapes should be omitted, false if they should be included. \p omitHiddenShapes defaults to true.
        @param containedMode false if only shapes that are within or interesect with the outline should be included, true if only shapes that are fully contained within the outline should be included. \p containedMode defaults to false
        @return returns a list of shapes."""
    def createGroupShape(self, name: str, shapes: List["Shape"]) -> "Shape":
        # type: (name, shapes) -> Shape:
        """@access public Q_SLOTS
         @brief createGroupShape combine a list of top level shapes into a group.
        @param name the name of the shape.
        @param shapes list of top level shapes.
        @return if successful, a GroupShape object will be returned."""

class TransparencyMask(Node):
    """* @brief The TransparencyMask class A transparency mask is a mask type node that can be used to show and hide parts of a layer."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return transparencymask If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def selection(self) -> "Selection":
        # type: () -> Selection:
        """@access public Q_SLOTS"""
    def setSelection(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS"""

class TransformMask(Node):
    """* @brief The TransformMask class A transform mask is a mask type node that can be used to store transformations."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return transformmask If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def toXML(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief toXML
        @return a string containing XML formated transform parameters."""
    def fromXML(self, xml: str) -> bool:
        # type: (xml) -> bool:
        """@access public Q_SLOTS
         @brief fromXML set the transform of the transform mask from XML formatted data. The xml must have a valid id dumbparams - placeholder for static transform masks tooltransformparams - static transform mask animatedtransformparams - animated transform mask
        @code
        <!DOCTYPE transform_params>
        <transform_params>
          <main id="tooltransformparams"/>
          <data mode="0">
           <free_transform>
            <transformedCenter type="pointf" x="12.3102137276208" y="11.0727768562035"/>
            <originalCenter type="pointf" x="20" y="20"/>
            <rotationCenterOffset type="pointf" x="0" y="0"/>
            <transformAroundRotationCenter value="0" type="value"/>
            <aX value="0" type="value"/>
            <aY value="0" type="value"/>
            <aZ value="0" type="value"/>
            <cameraPos z="1024" type="vector3d" x="0" y="0"/>
            <scaleX value="1" type="value"/>
            <scaleY value="1" type="value"/>
            <shearX value="0" type="value"/>
            <shearY value="0" type="value"/>
            <keepAspectRatio value="0" type="value"/>
            <flattenedPerspectiveTransform m23="0" m31="0" m32="0" type="transform" m33="1" m12="0" m13="0" m22="1" m11="1" m21="0"/>
            <filterId value="Bicubic" type="value"/>
           </free_transform>
          </data>
        </transform_params>
        @endcode
        @param xml a valid formated XML string with proper main and data elements.
        @return a true response if successful, a false response if failed."""

class Swatch:
    """* @brief The Swatch class is a thin wrapper around the KisSwatch class. A Swatch is a single color that is part of a palette, that has a name and an id. A Swatch color can be a spot color."""

class SelectionMask(Node):
    """* @brief The SelectionMask class A selection mask is a mask type node that can be used to store selections. In the gui, these are referred to as local selections. A selection mask can hold both raster and vector selections, though the API only supports raster selections."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return selectionmask If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def selection(self) -> "Selection":
        # type: () -> Selection:
        """@access public Q_SLOTS"""
    def setSelection(self, selection: "Selection") -> None:
        # type: (selection) -> None:
        """@access public Q_SLOTS"""

class Scratchpad(QWidget):
    """* @brief The Scratchpad class A scratchpad is a type of blank canvas area that can be painted on  with the normal painting devices"""

    def clear(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief Clears out scratchpad with color specfified set during setup"""
    def setFillColor(self, color: QColor) -> None:
        # type: (color) -> None:
        """@access public Q_SLOTS
         @brief Fill the entire scratchpad with a color
        @param Color to fill the canvas with"""
    def setModeManually(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief Switches between a GUI controlling the current mode and when mouse clicks control mode
        @param Setting to true allows GUI to control the mode with explicitly setting mode"""
    def setMode(self, modeName: str) -> None:
        # type: (modeName) -> None:
        """@access public Q_SLOTS
         @brief Manually set what mode scratchpad is in. Ignored if "setModeManually is set to false
        @param Available options are: "painting", "panning", and "colorsampling" """
    def linkCanvasZoom(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief Makes a connection between the zoom of the canvas and scratchpad area so they zoom in sync
        @param Should the scratchpad share the zoom level. Default is true"""
    def loadScratchpadImage(self, image: QImage) -> None:
        # type: (image) -> None:
        """@access public Q_SLOTS
         @brief Load image data to the scratchpad
        @param Image object to load"""
    def copyScratchpadImageData(self) -> QImage:
        # type: () -> QImage:
        """@access public Q_SLOTS
         @brief Take what is on the scratchpad area and grab image
        @return the image data from the scratchpage"""

class PresetChooser(KisPresetChooser):
    """* @brief The PresetChooser widget wraps the KisPresetChooser widget. The widget provides for selecting brush presets. It has a tagging bar and a filter field. It is not automatically synchronized with  the currently selected preset in the current Windows."""

    def setCurrentPreset(self, resource: "Resource") -> None:
        # type: (resource) -> None:
        """@access public Q_SLOTS
        Make the given preset active."""
    def currentPreset(self) -> "Resource":
        # type: () -> Resource:
        """@access public Q_SLOTS
        @return a Resource wrapper around the currently selected preset."""
    def presetSelected(self, resource: "Resource") -> None:
        # type: (resource) -> None:
        """@access public Q_SLOTS
        Emitted whenever a user selects the given preset."""
    def presetClicked(self, resource: "Resource") -> None:
        # type: (resource) -> None:
        """@access public Q_SLOTS
        Emitted whenever a user clicks on the given preset."""

class PaletteView(QWidget):
    """* @class PaletteView
    @brief The PaletteView class is a wrapper around a MVC method for handling palettes. This class shows a nice widget that can drag and drop, edit colors in a colorset and will handle adding and removing entries if you'd like it to.
    """

    def setPalette(self, palette: "Palette") -> None:
        # type: (palette) -> None:
        """@access public Q_SLOTS
         @brief setPalette Set a new palette.
        @param palette"""
    def addEntryWithDialog(self, color: "ManagedColor") -> bool:
        # type: (color) -> bool:
        """@access public Q_SLOTS
         @brief addEntryWithDialog This gives a simple dialog for adding colors, with options like adding name, id, and to which group the color should be added.
        @param color the default color to add
        @return whether it was successful."""
    def addGroupWithDialog(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief addGroupWithDialog gives a little dialog to ask for the desired groupname.
        @return whether this was successful."""
    def removeSelectedEntryWithDialog(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief removeSelectedEntryWithDialog removes the selected entry. If it is a group, it pop up a dialog asking whether the colors should also be removed.
        @return whether this was successful"""
    def trySelectClosestColor(self, color: "ManagedColor") -> None:
        # type: (color) -> None:
        """@access public Q_SLOTS
         @brief trySelectClosestColor tries to select the closest color to the one given. It does not force a change on the active color.
        @param color the color to compare to."""
    def entrySelectedForeGround(self, entry: "Swatch") -> None:
        # type: (entry) -> None:
        """@access public Q_SLOTS
         @brief entrySelectedForeGround fires when a swatch is selected with leftclick.
        @param entry"""
    def entrySelectedBackGround(self, entry: "Swatch") -> None:
        # type: (entry) -> None:
        """@access public Q_SLOTS
         @brief entrySelectedBackGround fires when a swatch is selected with rightclick.
        @param entry"""

class GroupShape(Shape):
    """* @brief The GroupShape class A group shape is a vector object with child shapes."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type returns the type.
        @return "groupshape" """
    def children(self) -> List["Shape"]:
        # type: () -> List[Shape]:
        """@access public Q_SLOTS
         @brief children
        @return the child shapes of this group shape."""

class GroupLayer(Node):
    """* @brief The GroupLayer class A group layer is a layer that can contain other layers. In Krita, layers within a group layer are composited first before they are added into the composition code for where the group is in the stack. This has a significant effect on how it is interpreted for blending modes. PassThrough changes this behaviour. Group layer cannot be animated, but can contain animated layers or masks."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return grouplayer"""
    def setPassThroughMode(self, passthrough: bool) -> None:
        # type: (passthrough) -> None:
        """@access public Q_SLOTS
         @brief setPassThroughMode This changes the way how compositing works. Instead of compositing all the layers before compositing it with the rest of the image, the group layer becomes a sort of formal way to organise everything. Passthrough mode is the same as it is in photoshop, and the inverse of SVG's isolation attribute(with passthrough=false being the same as isolation="isolate").
        @param passthrough whether or not to set the layer to passthrough."""
    def passThroughMode(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief passThroughMode
        @return returns whether or not this layer is in passthrough mode. @see setPassThroughMode"""

class FilterMask(Node):
    """* @brief The FilterMask class A filter mask, unlike a filter layer, will add a non-destructive filter to the composited image of the node it is attached to. You can set grayscale pixeldata on the filter mask to adjust where the filter is applied. Filtermasks can be animated."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return The type of the node. Valid types are: <ul>  <li>paintlayer  <li>grouplayer  <li>filelayer  <li>filterlayer  <li>filllayer  <li>clonelayer  <li>vectorlayer  <li>transparencymask  <li>filtermask  <li>transformmask  <li>selectionmask  <li>colorizemask </ul> If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def setFilter(self, filter: "Filter") -> None:
        # type: (filter) -> None:
        """@access public Q_SLOTS"""

class FilterLayer(Node):
    """* @brief The FilterLayer class A filter layer will, when compositing, take the composited image up to the point of the loction of the filter layer in the stack, create a copy and apply a filter. This means you can use blending modes on the filter layers, which will be used to blend the filtered image with the original. Similarly, you can activate things like alpha inheritance, or you can set grayscale pixeldata on the filter layer to act as a mask. Filter layers can be animated."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return "filterlayer" """
    def setFilter(self, filter: "Filter") -> None:
        # type: (filter) -> None:
        """@access public Q_SLOTS"""

class FillLayer(Node):
    """* @brief The FillLayer class A fill layer is much like a filter layer in that it takes a name and filter. It however specializes in filters that fill the whole canvas, such as a pattern or full color fill."""

    def __init__(self, filter: "Filter") -> None:
        """ """
    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return The type of the node. Valid types are: <ul>  <li>paintlayer  <li>grouplayer  <li>filelayer  <li>filterlayer  <li>filllayer  <li>clonelayer  <li>vectorlayer  <li>transparencymask  <li>filtermask  <li>transformmask  <li>selectionmask  <li>colorizemask </ul> If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def setGenerator(self, generatorName: str, filterConfig: "InfoObject") -> bool:
        # type: (generatorName, filterConfig) -> bool:
        """@access public Q_SLOTS
         @brief setGenerator set the given generator for this fill layer
        @param generatorName "pattern" or "color"
        @param filterConfig a configuration object appropriate to the given generator plugin
        @return true if the generator was correctly created and set on the layer"""

class FileLayer(Node):
    """* @brief The FileLayer class A file layer is a layer that can reference an external image and show said reference in the layer stack. If the external image is updated, Krita will try to update the file layer image as well."""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return "filelayer" """
    def setProperties(self, fileName: str, scalingMethod: str = str("None")) -> None:
        # type: (fileName, scalingMethod) -> None:
        """@access public Q_SLOTS
         @brief setProperties Change the properties of the file layer.
        @param fileName - A String containing the absolute file name.
        @param scalingMethod - a string with the scaling method, defaults to "None",  other options are "ToImageSize" and "ToImagePPI"
        """
    def resetCache(self) -> None:
        # type: () -> None:
        """@access public Q_SLOTS
        @brief makes the file layer to reload the connected image from disk"""
    def path(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief path
        @return A QString with the full path of the referenced image."""
    def scalingMethod(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief scalingMethod returns how the file referenced is scaled.
        @return one of the following: <ul>  <li> None - The file is not scaled in any way.  <li> ToImageSize - The file is scaled to the full image size;  <li> ToImagePPI - The file is scaled by the PPI of the image. This keep the physical dimensions the same. </ul>
        """
    def getFileNameFromAbsolute(self, basePath: str, filePath: str) -> str:
        # type: (basePath, filePath) -> str:
        """@access private
         @brief getFileNameFromAbsolute referenced from the fileLayer dialog, this will jumps through all the hoops to ensure that an appropriate filename will be gotten.
        @param baseName the location of the document.
        @param absolutePath the absolute location of the file referenced.
        @return the appropriate relative path."""

class DockWidgetFactoryBase(KoDockFactoryBase):
    """* @brief The DockWidgetFactoryBase class is the base class for plugins that want to add a dock widget to every window. You do not need to implement this class yourself, but create a DockWidget implementation and then add the DockWidgetFactory to the Krita instance like this:
    @code
    class HelloDocker(DockWidget):
      def __init__(self):
          super().__init__()
          label = QLabel("Hello", self)
          self.setWidget(label)
          self.label = label
    def canvasChanged(self, canvas):
          self.label.setText("Hellodocker: canvas changed");
    Application.addDockWidgetFactory(DockWidgetFactory("hello", DockWidgetFactoryBase.DockRight, HelloDocker))
    @endcode"""

class DockWidget(QDockWidget):
    """* DockWidget is the base class for custom Dockers. Dockers are created by a factory class which needs to be registered by calling Application.addDockWidgetFactory:
    @code
    class HelloDocker(DockWidget):
      def __init__(self):
          super().__init__()
          label = QLabel("Hello", self)
          self.setWidget(label)
          self.label = label
          self.setWindowTitle("Hello Docker")
    def canvasChanged(self, canvas):
          self.label.setText("Hellodocker: canvas changed");
    Application.addDockWidgetFactory(DockWidgetFactory("hello", DockWidgetFactoryBase.DockRight, HelloDocker))
    @endcode One docker per window will be created, not one docker per canvas or view. When the user switches between views/canvases, canvasChanged will be called. You can override that method to reset your docker's internal state, if necessary.
    """

    def canvas(self) -> "Canvas":
        # type: () -> Canvas:
        """@access public
        @@return the canvas object that this docker is currently associated with"""
    def canvasChanged(self, canvas: "Canvas") -> None:
        # type: (canvas) -> None:
        """@access public
         @brief canvasChanged is called whenever the current canvas is changed in the mainwindow this dockwidget instance is shown in.
        @param canvas The new canvas."""

class ColorizeMask(Node):
    """* @brief The ColorizeMask class A colorize mask is a mask type node that can be used to color in line art.
    @code
    window = Krita.instance().activeWindow()
    doc = Krita.instance().createDocument(10, 3, "Test", "RGBA", "U8", "", 120.0)
    window.addView(doc)
    root = doc.rootNode();
    node = doc.createNode("layer", "paintLayer")
    root.addChildNode(node, None)
    nodeData = QByteArray.fromBase64(b"AAAAAAAAAAAAAAAAEQYMBhEGDP8RBgz/EQYMAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARBgz5EQYM/xEGDAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEQYMAhEGDAkRBgwCAAAAAAAAAAAAAAAA");
    node.setPixelData(nodeData,0,0,10,3)

    cols = [ ManagedColor('RGBA','U8',''), ManagedColor('RGBA','U8','') ]
    cols[0].setComponents([0.65490198135376, 0.345098048448563, 0.474509805440903, 1.0]);
    cols[1].setComponents([0.52549022436142, 0.666666686534882, 1.0, 1.0]);
    keys = [
            QByteArray.fromBase64(b"/48AAAAAAAAAAAAAAAAAAAAAAACmCwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            QByteArray.fromBase64(b"AAAAAAAAAACO9ocAAAAAAAAAAAAAAAAAAAAAAMD/uQAAAAAAAAAAAAAAAAAAAAAAGoMTAAAAAAAAAAAA")
            ]

    mask = doc.createColorizeMask('c1')
    node.addChildNode(mask,None)
    mask.setEditKeyStrokes(True)

    mask.setUseEdgeDetection(True)
    mask.setEdgeDetectionSize(4.0)
    mask.setCleanUpAmount(70.0)
    mask.setLimitToDeviceBounds(True)
    mask.initializeKeyStrokeColors(cols)

    for col,key in zip(cols,keys):
        mask.setKeyStrokePixelData(key,col,0,0,20,3)

    mask.updateMask()
    mask.setEditKeyStrokes(False);
    mask.setShowOutput(True);
    @endcode"""

    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return colorizemask If the Node object isn't wrapping a valid Krita layer or mask object, and empty string is returned.
        """
    def keyStrokesColors(self) -> List["ManagedColor"]:
        # type: () -> List[ManagedColor]:
        """@access public Q_SLOTS
         @brief keyStrokesColors Colors used in the Colorize Mask's keystrokes.
        @return a ManagedColor list containing the colors of keystrokes."""
    def initializeKeyStrokeColors(
        self, colors: List["ManagedColor"], transparentIndex: int = -1
    ) -> None:
        # type: (colors, transparentIndex) -> None:
        """@access public Q_SLOTS
         @brief initializeKeyStrokeColors Set the colors to use for the Colorize Mask's keystrokes.
        @param colors a list of ManagedColor to use for the keystrokes.
        @param transparentIndex index of the color that should be marked as transparent."""
    def removeKeyStroke(self, color: "ManagedColor") -> None:
        # type: (color) -> None:
        """@access public Q_SLOTS
         @brief removeKeyStroke Remove a color from the Colorize Mask's keystrokes.
        @param color a ManagedColor to be removed from the keystrokes."""
    def transparencyIndex(self) -> int:
        # type: () -> int:
        """@access public Q_SLOTS
         @brief transparencyIndex Index of the transparent color.
        @return an integer containing the index of the current color marked as transparent."""
    def keyStrokePixelData(
        self, color: "ManagedColor", x: int, y: int, w: int, h: int
    ) -> QByteArray:
        # type: (color, x, y, w, h) -> QByteArray:
        """@access public Q_SLOTS
         @brief keyStrokePixelData reads the given rectangle from the keystroke image data and returns it as a byte array. The pixel data starts top-left, and is ordered row-first.
        @param color a ManagedColor to get keystrokes pixeldata from.
        @param x x position from where to start reading
        @param y y position from where to start reading
        @param w row length to read
        @param h number of rows to read
        @return a QByteArray with the pixel data. The byte array may be empty."""
    def setKeyStrokePixelData(
        self,
        value: Union[QByteArray, bytes, bytearray],
        color: "ManagedColor",
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> bool:
        # type: (value, color, x, y, w, h) -> bool:
        """@access public Q_SLOTS
         @brief setKeyStrokePixelData writes the given bytes, of which there must be enough, into the keystroke, the keystroke's original pixels are overwritten
        @param value the byte array representing the pixels. There must be enough bytes available. Krita will take the raw pointer from the QByteArray and start reading, not stopping before (number of channels * size of channel * w * h) bytes are read.
        @param color a ManagedColor to set keystrokes pixeldata for.
        @param x the x position to start writing from
        @param y the y position to start writing from
        @param w the width of each row
        @param h the number of rows to write
        @return true if writing the pixeldata worked"""
    def setUseEdgeDetection(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setUseEdgeDetection Activate this for line art with large solid areas, for example shadows on an object.
        @param value true to enable edge detection, false to disable."""
    def useEdgeDetection(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief useEdgeDetection
        @return true if Edge detection is enabled, false if disabled."""
    def setEdgeDetectionSize(self, value: float) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setEdgeDetectionSize Set the value to the thinnest line on the image.
        @param value a float value of the edge size to detect in pixels."""
    def edgeDetectionSize(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
         @brief edgeDetectionSize
        @return a float value of the edge detection size in pixels."""
    def setCleanUpAmount(self, value: float) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setCleanUpAmount This will attempt to handle messy strokes that overlap the line art where they shouldn't.
        @param value a float value from 0.0 to 100.00 where 0.0 is no cleanup is done and 100.00 is most aggressive.
        """
    def cleanUpAmount(self) -> float:
        # type: () -> float:
        """@access public Q_SLOTS
         @brief cleanUpAmount
        @return a float value of 0.0 to 100.0 represening the cleanup amount where 0.0 is no cleanup is done and 100.00 is most aggressive.
        """
    def setLimitToDeviceBounds(self, value: bool) -> None:
        # type: (value) -> None:
        """@access public Q_SLOTS
         @brief setLimitToDeviceBounds Limit the colorize mask to the combined layer bounds of the strokes and the line art it is filling. This can speed up the use of the mask on complicated compositions, such as comic pages.
        @param value set true to enabled limit bounds, false to disable."""
    def limitToDeviceBounds(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief limitToDeviceBounds
        @return true if limit bounds is enabled, false if disabled."""
    def updateMask(self, force: bool = False) -> None:
        # type: (force) -> None:
        """@access public Q_SLOTS
         @brief updateMask Process the Colorize Mask's keystrokes and generate a projection of the computed colors.
        @param force force an update"""
    def showOutput(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief showOutput Show output mode allows the user to see the result of the Colorize Mask's algorithm.
        @return true if edit show coloring mode is enabled, false if disabled."""
    def setShowOutput(self, enabled: bool) -> None:
        # type: (enabled) -> None:
        """@access public Q_SLOTS
         @brief setShowOutput Toggle Colorize Mask's show output mode.
        @param enabled set true to enable show coloring mode and false to disable it."""
    def editKeyStrokes(self) -> bool:
        # type: () -> bool:
        """@access public Q_SLOTS
         @brief editKeyStrokes Edit keystrokes mode allows the user to modify keystrokes on the active Colorize Mask.
        @return true if edit keystrokes mode is enabled, false if disabled."""
    def setEditKeyStrokes(self, enabled: bool) -> None:
        # type: (enabled) -> None:
        """@access public Q_SLOTS
         @brief setEditKeyStrokes Toggle Colorize Mask's edit keystrokes mode.
        @param enabled set true to enable edit keystrokes mode and false to disable it."""

class CloneLayer(Node):
    """* @brief The CloneLayer class A clone layer is a layer that takes a reference inside the image and shows the exact same pixeldata. If the original is updated, the clone layer will update too."""

    def __init__(self, enabled: bool) -> None:
        """@brief setEditKeyStrokes Toggle Colorize Mask's edit keystrokes mode.
        @param enabled set true to enable edit keystrokes mode and false to disable it."""
    def type(self) -> str:
        # type: () -> str:
        """@access public Q_SLOTS
         @brief type Krita has several types of nodes, split in layers and masks. Group layers can contain other layers, any layer can contain masks.
        @return clonelayer"""
    def sourceNode(self) -> "Node":
        # type: () -> Node:
        """@access public Q_SLOTS
         @brief sourceNode
        @return the node the clone layer is based on."""
    def setSourceNode(self, node: "Node") -> None:
        # type: (node) -> None:
        """@access public Q_SLOTS
         @brief setSourceNode
        @param node the node to use as the source of the clone layer."""
