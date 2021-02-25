$(document).ready(function() {

    var dropContainer = document.getElementById('drop-container');
    dropContainer.ondragover = dropContainer.ondragend = function() {
      return false;
    };
  
    dropContainer.ondrop = function(e) {
      e.preventDefault();
      loadImage(e.dataTransfer.files[0])
    }
  
    $("#browse-button").change(function() {
      loadImage($("#browse-button").prop("files")[0]);
    });
  
    $('.modal').modal({
      dismissible: false,
      ready: function(modal, trigger) {
        $.ajax({
          type: "POST",
          url: '/detect/api/',
          data: {
            'image64': $('#img-card-1').attr('src')
          },
          dataType: 'text',
          success: function(data) {
            loadStats(data)
          }
        }).always(function() {
          modal.modal('close');
        });
      }
    });
   
    $('#go-back').click(function() {
      $('#img-card-1').removeAttr("src");
      $('#stat-table').html('');
      switchCard(0);
    });
    $('#go-start').click(function() {
      var elem = document.getElementById("img-card-2");
      elem.parentNode.removeChild(elem);
      $('#stat-table').html('');
      switchCard(0);
    });
  
    $('#upload-button').click(function() {
      $('.modal').modal('open');
    });
  });
  
  switchCard = function(cardNo) {
    var containers = [".dd-container", ".uf-container", ".dt-container"];
    var visibleContainer = containers[cardNo];
    for (var i = 0; i < containers.length; i++) {
      var oz = (containers[i] === visibleContainer) ? '1' : '0';
      $(containers[i]).animate({
        opacity: oz
      }, {
        duration: 200,
        queue: false,
      }).css("z-index", oz);
    }
  }
  
  loadImage = function(file) {
    var reader = new FileReader();
    reader.onload = function(event) {
      $('#img-card-1').attr('src', event.target.result);
    }
    reader.readAsDataURL(file);
    switchCard(1);  
  }
  
  loadStats = function(jsonData) {
    switchCard(2);
    var data = JSON.parse(jsonData);
    if(data["success"] == true){
      var elem = document.createElement("img");
      elem.setAttribute('class', "card crop");
      elem.setAttribute('id', 'img-card-2');
      elem.src = data['url'];
      document.getElementById("result-image").appendChild(elem);
      // var yolo4 = document.getElementById('yolo4_prediction');
      // yolo4.innerHTML = data['yolo4_prediction'];
      // var eff = document.getElementById('eff_prediction');
      // eff.innerHTML = data['eff_prediction'];
      buildHtmlTable(data['yolo4_prediction'],'#excelDataTableyolo4_prediction')
      buildHtmlTable(data['eff_prediction'],'#excelDataTableeff_prediction')
      
    }
  }
  function buildHtmlTable(myList, selector) {
    console.log(myList)
    myList = JSON.parse(myList);
    var columns = addAllColumnHeaders(myList, selector);
  
    for (var i = 0; i < myList.length; i++) {
      var row$ = $('<tr/>');
      for (var colIndex = 0; colIndex < columns.length; colIndex++) {
        var cellValue = myList[i][columns[colIndex]];
        if (cellValue == null) cellValue = "";
        row$.append($('<td/>').html(cellValue));
      }
      $(selector).append(row$);
    }
  }
  
  // Adds a header row to the table and returns the set of columns.
  // Need to do union of keys from all records as some records may not contain
  // all records.
  function addAllColumnHeaders(myList, selector) {
    var columnSet = [];
    var headerTr$ = $('<tr/>');
  
    for (var i = 0; i < myList.length; i++) {
      var rowHash = myList[i];
      for (var key in rowHash) {
        if ($.inArray(key, columnSet) == -1) {
          columnSet.push(key);
          headerTr$.append($('<th/>').html(key));
        }
      }
    }
    $(selector).append(headerTr$);
  
    return columnSet;
  }