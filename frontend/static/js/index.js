$('#startup_button').click(function() {
    $.ajax({
        url: '/api/start_all/',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({"111": "222", "333": "444"}), // 发送的数据
        success: function(data) {
            console.log('Success:', data);
        },
        error: function(error) {
            console.error('Error:', error);
        }
    });
});