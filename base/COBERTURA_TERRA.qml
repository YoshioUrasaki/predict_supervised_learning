<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.10.8-A Coruña" maxScale="0" minScale="1e+08" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <rasterrenderer opacity="1" classificationMax="7" type="singlebandpseudocolor" classificationMin="1" alphaBand="-1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader clip="0" colorRampType="INTERPOLATED" classificationMode="2">
          <colorramp type="gradient" name="[source]">
            <prop k="color1" v="215,25,28,255"/>
            <prop k="color2" v="44,123,182,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.25;253,174,97,255:0.5;255,255,191,255:0.75;171,217,233,255"/>
          </colorramp>
          <item label="1" value="1" color="#83e462" alpha="255"/>
          <item label="2" value="2" color="#fffecb" alpha="255"/>
          <item label="3" value="3" color="#d590b0" alpha="255"/>
          <item label="4" value="4" color="#f6cece" alpha="255"/>
          <item label="5" value="5" color="#8cffee" alpha="255"/>
          <item label="6" value="6" color="#39adeb" alpha="255"/>
          <item label="7" value="7" color="#ffffff" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0"/>
    <huesaturation colorizeOn="0" colorizeGreen="128" colorizeBlue="128" colorizeRed="255" saturation="0" colorizeStrength="100" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
