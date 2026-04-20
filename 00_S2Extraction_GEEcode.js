var assetPath = 'projects/mythic-inn-476207-j1/assets/Lower_chi_Subbasin';
var myShapefile = ee.FeatureCollection(assetPath);
var yasGeom = myShapefile.geometry();

var utm48   = ee.Projection('EPSG:32648');
var rect    = yasGeom.transform(utm48, 1).bounds(1, utm48).transform('EPSG:4326', 1);

var bandsToExport = ['B2', 'B3', 'B4', 'B8'];

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2022-10-31', '2023-04-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .filterBounds(rect)
  .select(bandsToExport)
  .sort('system:time_start');

// ดึงรายชื่อ Metadata มาทีเดียว (ป้องกัน Chrome Unresponsive)
var s2Info = s2.reduceColumns(ee.Reducer.toList(2), ['system:time_start', 'MGRS_TILE'])
               .get('list').getInfo();

print('กำลังเตรียม Export ทั้งหมด: ' + (s2Info.length * bandsToExport.length) + ' tasks...');

s2Info.forEach(function(info) {
  var timestamp = info[0];
  var tile = info[1] || 'UnknownTile';
  
  // แปลงเวลาเป็น Format วันที่ (JavaScript side)
  var date = new Date(timestamp);
  var dateStr = date.toISOString().split('T')[0].replace(/-/g, '_');
  
  // ดึงภาพ ณ เวลานั้นๆ ออกมา
  var img = s2.filter(ee.Filter.eq('system:time_start', timestamp)).first();
  
  // วนลูปแยกแบนด์เพื่อส่ง Export ทีละไฟล์
  bandsToExport.forEach(function(band) {
    var desc = 'S2_' + dateStr + '_' + tile + '_' + band;
    
    Export.image.toDrive({
      image: img.select(band),
      description: desc,
      folder: 'S2_Lower_Chi_2022_BandsALL', // เปลี่ยนชื่อโฟลเดอร์ให้ชัดเจน
      fileFormat: 'GeoTIFF',
      region: rect,
      scale: 10,
      crs: 'EPSG:32648',
      maxPixels: 1e13
    });
  });
});
