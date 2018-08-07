<?xml version="1.0"?>
<!-- Author: Mauricio Villegas <mauricio_ville@yahoo.com> -->
<xsl:stylesheet
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xmlns:str="http://exslt.org/strings"
xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"
xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"
extension-element-prefixes="str"
version="1.0">

<xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

<xsl:template match="@* | node()">
  <xsl:copy>
    <xsl:apply-templates select="@* | node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="//_:Coords">
  <Coords>
    <xsl:attribute name="points">
      <xsl:for-each select="_:Point">
        <xsl:choose>
          <xsl:when test="position() = 1">
            <xsl:value-of select="concat(@x,',',@y)"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="concat(' ',@x,',',@y)"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:for-each>
    </xsl:attribute>
  </Coords>
</xsl:template>

</xsl:stylesheet>
