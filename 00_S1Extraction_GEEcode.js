// =====================================================================
// Sentinel-1: 5-Day Composite Export (Separate Bands)
// Compatible with 00_CropS1.py
// =====================================================================

var assetPath = 'projects/mythic-inn-476207-j1/assets/Lower_chi_Subbasin';
var myShapefile = ee.FeatureCollection(assetPath);
var yasGeom = myShapefile.geometry();

var utm48   = ee.Projection('EPSG:32648');
var rect    = yasGeom.transform(utm48, 1).bounds(1, utm48).transform('EPSG:4326', 1);

Map.addLayer(yasGeom, {color: 'cyan'}, 'Lower Chi Subbasin');
Map.centerObject(yasGeom, 10);

var startDate = '2022-09-12';
var endDate = '2022-09-30';

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate(startDate, endDate)
  .filterBounds(rect)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('productType', 'GRD'))
  .sort('system:time_start');

var nDays = ee.Date(endDate).difference(ee.Date(startDate), 'day').round();

// ขยับทีละ 5 วันเช่นเดียวกัน
var dayOffsets = ee.List.sequence(0, nDays, 5); 

var dates5Days = dayOffsets.map(function(offset) {
  return ee.Date(startDate).advance(offset, 'day').format('YYYY_MM_dd');
});

var dateList = dates5Days.getInfo(); 
print('Total 5-Day periods found (S1):', dateList.length);

function get5DayComposite(dateStr, bandName) {
  var d0 = ee.Date.parse('YYYY_MM_dd', dateStr);
  var d1 = d0.advance(5, 'day'); // 5-Day Window
  
  var windowCol = s1.filterDate(d0, d1).filter(ee.Filter.listContains('transmitterReceiverPolarisation', bandName));
  
  // แปลงค่าเป็น Linear เพื่อหาค่าเฉลี่ยที่ถูกต้องในช่วง 5 วัน
  var linMean = windowCol.map(function(img) {
    var b = ee.Image(img).select(bandName);
    return ee.Image(10).pow(b.divide(10)).rename(bandName);
  }).mean();
  
  // แปลงกลับเป็นหน่วย dB และ Clip
  return linMean.log10().multiply(10).rename(bandName).clip(rect);
}

dateList.forEach(function(dateStr) {
  ['VV', 'VH'].forEach(function(band) {
    var finalImg = get5DayComposite(dateStr, band);
    
    // 🌟 ชื่อไฟล์แบบ S1_YYYY_MM_DD_POL เข้ากับ 00_CropS1.py ทันที
    var desc = 'S1_' + dateStr + '_' + band; 
    
    Export.image.toDrive({
      image: finalImg,
      description: desc,
      folder: 'S1_Lower_Chi_2022ALL9', // โฟลเดอร์ที่ 00_CropS1 คาดหวัง
      fileFormat: 'GeoTIFF',
      region: rect, 
      scale: 10,
      crs: 'EPSG:32647', // ใช้ UTM 47N ตามเดิมของคุณ
      maxPixels: 1e13
    });
  });
});
