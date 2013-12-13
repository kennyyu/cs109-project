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
				$("#comment-result").html(parseInt(data));
			},
		});
	});

	$("#sentence-button").click(function(){
		var old = $("#word").val();
		var word = old.split(' ').pop();
		$.ajax({
			url: SENTENCE_ENDPOINT,
			type: "GET",
			data: {
				"word": word
			},
			success: function(data, status, xhr) {
                if (old != '' && data == '' && old[old.length - 1] != '.')
                    $("#word").val(old + '.');
                else
                    $("#word").val($.trim(old + ' ' + data));
			},
		});
	});

	$("#clear-button").click(function(){
		$("#word").val("");
	});

});
