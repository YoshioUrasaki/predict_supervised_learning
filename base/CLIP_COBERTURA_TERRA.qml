<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" maxScale="0" minScale="1e+08" hasScaleBasedVisibilityFlag="0" version="3.10.11-A Coruña">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>0</Searchable>
  </flags>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <rasterrenderer alphaBand="-1" band="1" classificationMin="1" opacity="1" type="singlebandpseudocolor" classificationMax="7">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="2" clip="0">
          <colorramp name="[source]" type="gradient">
            <prop k="color1" v="215,25,28,255"/>
            <prop k="color2" v="44,123,182,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.25;253,174,97,255:0.5;255,255,191,255:0.75;171,217,233,255"/>
          </colorramp>
          <item color="#83e462" alpha="255" label="1" value="1"/>
          <item color="#fffecb" alpha="255" label="2" value="2"/>
          <item color="#d590b0" alpha="255" label="3" value="3"/>
          <item color="#f6cece" alpha="255" label="4" value="4"/>
          <item color="#8cffee" alpha="255" label="5" value="5"/>
          <item color="#39adeb" alpha="255" label="6" value="6"/>
          <item color="#ffffff" alpha="255" label="7" value="7"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0"/>
    <huesaturation colorizeRed="255" colorizeGreen="128" saturation="0" colorizeOn="0" colorizeStrength="100" colorizeBlue="128" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
