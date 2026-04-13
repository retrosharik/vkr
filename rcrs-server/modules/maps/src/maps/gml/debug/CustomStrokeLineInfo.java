package maps.gml.debug;

import rescuecore2.misc.geometry.Line2D;
import rescuecore2.misc.geometry.Point2D;
import rescuecore2.misc.gui.ScreenTransform;
import rescuecore2.misc.gui.ShapeDebugFrame;

import java.awt.*;
import java.awt.geom.Path2D;
import java.util.Collections;

/**
 * A ShapeInfo for drawing a Line2D with a custom stroke (e.g. dashed or thick)
 * This class inherits from Line2DShapeInfo to reuse its basic structure
 * but overrides the paint method to apply a custom stroke.
 */
public class CustomStrokeLineInfo extends ShapeDebugFrame.Line2DShapeInfo {
    private final Stroke stroke;
    private final Color colour;
    private final Line2D line;

    /**
     * Construct a new CustomStrokeLineInfo.
     * @param line   The line to display.
     * @param name   The name of the shape.
     * @param colour The colour of the shape.
     * @param stroke The custom stroke to use for drawing.
     */
    public CustomStrokeLineInfo(Line2D line, String name, Color colour, Stroke stroke) {
        // Call the parent constructor with dummy values for thickness and arrows,
        // as we will control drawing ourselves.
        super(Collections.singleton(line), name, colour, false, false);
        this.line = line;
        this.stroke = stroke;
        this.colour = colour;
    }

    @Override
    public Shape paint(Graphics2D g, ScreenTransform transform) {
        // This overrides the parent's paint method to apply our custom stroke

        // Save the original stroke to restore it later
        Stroke oldStroke = g.getStroke();

        try {
            g.setStroke(this.stroke);
            g.setColor(this.colour);

            Point2D start = line.getOrigin();
            Point2D end = line.getEndPoint();
            int x1 = transform.xToScreen(start.getX());
            int y1 = transform.yToScreen(start.getY());
            int x2 = transform.xToScreen(end.getX());
            int y2 = transform.yToScreen(end.getY());

            g.drawLine(x1, y1, x2, y2);

            // Return a shape for mouseover detection
            Path2D path = new Path2D.Double();
            path.moveTo(x1, y1);
            path.lineTo(x2, y2);
            return g.getStroke().createStrokedShape(path);
        } finally {
            // Always restore the original stroke
            g.setStroke(oldStroke);
        }
    }
}
