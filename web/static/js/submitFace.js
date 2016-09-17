function callEvaluate(image)
{
	var full = location.protocol+'//'+location.hostname+(location.port ? ':'+location.port: '');
	var split = image.split(full + '/')
	var imgLocation = split[1].toString()
	$.get({
		url: this.location.href,
		data: { img: imgLocation },
		success: function (data, textStatus, jqXHR) {
			$('#resultModal').html(data.toString());
			$('#resultModal').modal();
		}
	});
}
