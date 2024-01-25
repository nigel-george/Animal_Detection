$(document).ready(function () {
    // Selecting elements by their IDs
    let leaf = document.getElementById('leaf');
    let hill1 = document.getElementById('hill1');
    let hill4 = document.getElementById('hill4');
    let hill5 = document.getElementById('hill5');
    let text = document.getElementById('text'); // Assuming 'text' is the ID of an element

    // Variable to keep track of the scroll limit
    const scrollLimit = 500; // Set the scroll limit to 500 pixels, change this as needed

    // Adding a scroll event listener to the window
    window.addEventListener('scroll', function () {
        // Getting the current vertical scroll position
        let value = window.scrollY;

        // Limiting the scroll effect up to a certain point (scrollLimit)
        if (value <= scrollLimit) {
            // Changing styles based on scroll position for the parallax effect
            text.style.marginTop = value * 1.5 + 'px'; // Adjust the factor for desired effect
            leaf.style.top = value - 1.5 + 'px';
            leaf.style.left = value * 1.5 + 'px';
            hill5.style.left = value * 1.5 + 'px';
            hill4.style.left = value * -1.5 + 'px';
            hill1.style.top = value * 1 + 'px'; // Adjust the factor for desired effect
        } else {
            // If scroll limit is reached, remove the scroll event listener to stop further scrolling
            window.removeEventListener('scroll', function () {});
        }
    });

    $('#inputfile').on('change', function (e) {
        var reader = new FileReader();
        reader.onload = function (event) {
            $('#selectedImagePreview').attr('src', event.target.result);
        };
        reader.readAsDataURL(e.target.files[0]);
    });

    // Additional script for handling file size and type validation
    $('#uploadButton').on('click', function () {
        // Check if a file is selected
        if ($('#inputfile').get(0).files.length === 0) {
            alert('Please select an image before uploading.');
            return;
        }

        // Validate file size and type before submitting the form
        let fileSize = $('#inputfile')[0].files[0].size / 1024 / 1024; // in MB
        if (fileSize > 1) {
            alert('File is too big. Images more than 1MB are not allowed.');
            return;
        }

        let ext = $('#inputfile').val().split('.').pop().toLowerCase();
        if ($.inArray(ext, ['jpg', 'jpeg']) === -1) {
            alert('Only jpeg/jpg files are allowed!');
            return;
        }

        // Submit the form if validation passes
        $('#imageUploadForm').submit();
    });
});
function resformsubmit(){

    $('.resformbutton').submit();

}
function redirectWebcam() {
    // Directly navigate to the /upload_webcam endpoint
    window.location.href = '/upload_webcam';
    
  }