var UPVOTE_ENDPOINT = "http://localhost:18888/up";
var SENTENCE_ENDPOINT = "http://localhost:18888/sentence";

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

	$("#sentence-button").click(function(){
		var word = $("#word").val();
		var old = $("#sentence").html();
		if (old == "") {
			$("#sentence").html(word);
		}
		$.ajax({
			url: SENTENCE_ENDPOINT,
			type: "GET",
			data: {
				"word": word
			},
			success: function(data, status, xhr) {
				$("#word").val(data);
				var old = $("#sentence").html();
				$("#sentence").html(old + " " + data);
			},
		});
	});

});