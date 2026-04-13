package maps.convert.osm2gml;

import maps.convert.ConvertStep;

import java.util.Collection;

/**
 * This step processes the intersection graph and generates the
 * geometric polygon areas for each intersection
 */
public class GenerateIntersectionAreaStep extends ConvertStep {
    private final TemporaryMap map;

    public GenerateIntersectionAreaStep(TemporaryMap map) {
        this.map = map;
    }

    @Override
    public String getDescription() {
        return "Generating intersection areas";
    }

    @Override
    protected void step() {
        Collection<OSMIntersectionInfo> intersections = map.getOSMIntersectionInfo();
        double sizeOf1m = ConvertTools.sizeOf1Metre(map.getOSMMap());

        setProgressLimit(intersections.size());

        for (OSMIntersectionInfo next : intersections) {
            next.process(sizeOf1m);
            bumpProgress();
        }

        setStatus("Generated polygon areas for " + intersections.size() + " intersections");
    }
}
