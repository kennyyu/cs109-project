var UPVOTE_ENDPOINT = "http://localhost:18888/up";

$(document).ready(function(){

	$("#upvote-button").click(function(){
		var comment = $("#comment").val();
		$.ajax({
			url: UPVOTE_ENDPOINT,
			type: "GET",
			data: {
				"comment": comment
			},
			success: function(data, status, xhr) {
				$("#comment-result").html(data);
			},
		});
	});

});