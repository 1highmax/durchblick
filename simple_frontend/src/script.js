var map;
var marker;

document.addEventListener("DOMContentLoaded", function () {
  map = L.map('map').setView([-25.363, 131.044], 4);
  var API_TOKEN = "b3f456b2-ce12-4d0a-928d-f5347a2085cc";

  L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png?api_key=' + API_TOKEN, {
    maxZoom: 20,
    attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org">OpenMapTiles</a>, &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
  }).addTo(map);

  marker = L.marker([-25.363, 131.044]).addTo(map); // Initialize the marker with a default position

  document.getElementById('address-input').addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
      searchAddress(event.target.value);
    }
  });
});

function searchAddress(address) {
  var nominatimUrl = 'https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(address) + '&limit=1';

  fetch(nominatimUrl)
    .then(function (data) {
      if (data && data.length > 0) {
        var lat = parseFloat(data[0].lat);
        var lon = parseFloat(data[0].lon);
        // ...

        // Fetch the JPEG image from the external API
        fetchImageFromAPI(lat, lon);

        // Update the output fields
        document.getElementById('output-field-1').textContent = "Sample value 1";
        document.getElementById('output-field-2').textContent = "Sample value 2";
        document.getElementById('output-field-3').textContent = "Sample value 3";
      } else {
        alert('Address not found');
      }
    })
    .catch(function (error) {
      console.error('Error fetching address data:', error);
    });
}

function fetchImageFromAPI(lat, lon) {
  var imageUrl = 'https://your-image-api-url.com?lat=' + lat + '&lon=' + lon;
}