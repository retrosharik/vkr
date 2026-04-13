package maps.convert.osm2gml;

import lombok.Getter;
import lombok.Setter;
import rescuecore2.misc.geometry.Point2D;

import maps.osm.OSMNode;

import java.awt.geom.Area;
import java.awt.geom.Path2D;

import java.util.List;
import java.util.ArrayList;

/**
   Information about an OSM road.
*/
public class OSMRoadInfo implements OSMShape {
    @Setter @Getter private OSMNode from;
    @Setter @Getter private OSMNode to;
    @Getter private Point2D fromLeft;
    @Getter private Point2D toLeft;
    @Getter private Point2D fromRight;
    @Getter private Point2D toRight;
    private Area area;

    /**
       Create an OSMRoadInfo between two nodes.
       @param from The first OSMNode.
       @param to The second OSMNode.
    */
    public OSMRoadInfo(OSMNode from, OSMNode to) {
        this.from = from;
        this.to = to;
        area = null;
    }

    /**
       Set the point that is on the left side of this road at the "from" end.
       @param p The from-left corner point.
    */
    public void setFromLeft(Point2D p) {
        fromLeft = p;
        area = null;
    }

    /**
       Set the point that is on the right side of this road at the "from" end.
       @param p The from-right corner point.
    */
    public void setFromRight(Point2D p) {
        fromRight = p;
        area = null;
    }

    /**
       Set the point that is on the left side of this road at the "to" end.
       @param p The to-left corner point.
    */
    public void setToLeft(Point2D p) {
        toLeft = p;
        area = null;
    }

    /**
       Set the point that is on the right side of this road at the "to" end.
       @param p The to-right corner point.
    */
    public void setToRight(Point2D p) {
        toRight = p;
        area = null;
    }

    @Override
    public Area getArea() {
        if (area == null) {
            if (fromLeft == null || fromRight == null || toLeft == null || toRight == null) {
                return null;
            }
            Path2D.Double path = new Path2D.Double();
            path.moveTo(fromLeft.getX(), fromLeft.getY());
            path.lineTo(fromRight.getX(), fromRight.getY());
            path.lineTo(toRight.getX(), toRight.getY());
            path.lineTo(toLeft.getX(), toLeft.getY());
            path.closePath();
            area = new Area(path.createTransformedShape(null));
        }
        return area;
    }

    @Override
    public List<Point2D> getVertices() {
        List<Point2D> result = new ArrayList<>();
        result.add(fromLeft);
        result.add(fromRight);
        result.add(toRight);
        result.add(toLeft);
        return result;
    }

    @Override
    public String toString() {
        return "RoadInfo [" + fromLeft + ", " + fromRight + ", " + toRight + ", " + toLeft + "]";
    }
}
