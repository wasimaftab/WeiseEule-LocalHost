"use strict";
/* ----##### Import packages BELOW #####---- */
// import { createApp } from 'vue'
import tippy from 'tippy.js'
import 'tippy.js/dist/tippy.css'
import Swal from 'sweetalert2'
import flatpickr from "flatpickr";
import "flatpickr/dist/flatpickr.min.css";


let count_next = 0;
let count_prev = 0;
let count = 0;
let rerank_count = 0;

/* ----##### USER DEFINED FUNCTIONS START #####---- */
function createMessageContainerElements() {
	let chatArea = document.getElementById('chat-area');
	let messageContainer = document.getElementById('message-container');
	if (!messageContainer) {
		messageContainer = document.createElement('div');
		messageContainer.id = 'message-container';
		chatArea.insertBefore(messageContainer, chatArea.firstChild);
	}
	return { chatArea, messageContainer };
}


function createUserMessageElements() {
	let userMessageContainer = document.createElement('div');
	userMessageContainer.className = 'message-container user-message-container';

	let userIcon = document.createElement('div');
	userIcon.className = 'user-icon';
	userMessageContainer.appendChild(userIcon);

	let userMessage = document.createElement('div');
	userMessage.className = 'message user-message';

	return { userMessageContainer, userIcon, userMessage };
}


function createBotMessageElements() {
	let botMessageContainer = document.createElement('div');
	botMessageContainer.className = 'message-container bot-message-container';

	let botIcon = document.createElement('div');
	botIcon.className = 'bot-icon';
	botMessageContainer.appendChild(botIcon);

	let botMessage = document.createElement('div');
	botMessage.className = 'message bot-message';

	return { botMessageContainer, botIcon, botMessage };
}


async function sendVectorIds(vectorIds,
	namespace,
	template,
	userInput) {
	// let inputField = document.getElementById('user-input');
	// let userInput = inputField.value;
	console.log("namespace = " + namespace + " inside sendVectorIds()");
	var llm = document.getElementById('select_llm').value;
	var temp = parseFloat(document.getElementById('set_temp').value);
	var paper_id = document.getElementById('set_paper_id').value;
	console.log("paper_id = " + paper_id + " inside sendVectorIds()");
	var answer_per_paper = document.getElementById('select_answer_per_paper').value;
	var chunks_from_one_paper = document.getElementById('select_chunks_from_one_paper').value;
	var py_script_path = 'pycodes/get_answer_rerank.py';
	let isError = false;

	/* Here you should process the user input and generate a response */
	console.log("Calling runPythonScript API from sendVectorIds() " + ++rerank_count + " times!");
	const response = await fetch('/api/runPythonScript', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			py_script_path: py_script_path,
			vector_ids: vectorIds,
			llm: llm,
			temp: temp,
			namespace: namespace,
			query: userInput,
			template: template,
			paper_id: paper_id,
			answer_per_paper: answer_per_paper,
			chunks_from_one_paper: chunks_from_one_paper
		}),
	});
	let result = await response.json();
	/* 	result is an object with a single property 'result' which is a JSON string
	You need to parse that JSON string to get the actual result object: */
	console.log("Printing result below:");
	console.log(result);
	result = JSON.parse(result.result);
	console.log("Printing result.code below:");
	console.log(result.code);
	if (result.code == "error" || result.code == "failure") {
		Swal.fire({
			icon: 'error',
			title: "Error in chat completion",
			text: result.output,
			allowOutsideClick: false
		});
		isError = true;
	} else {
		// Hide datatable modal after backend python code has returned result
		$('#dataModal').hide();

		/* display user message on console */
		await displayChatResponse(result, userInput);

		// Clear the input field
		let inputField = document.getElementById('user-input');
		inputField.value = '';
	}
	return isError;
}


function displayChatResponse(result, userInput) {
	let { chatArea, messageContainer } = createMessageContainerElements();
	let { userMessageContainer, userIcon, userMessage } = createUserMessageElements();
	let { botMessageContainer, botIcon, botMessage } = createBotMessageElements();
	userMessage.textContent = userInput;
	userMessageContainer.appendChild(userMessage);
	messageContainer.appendChild(userMessageContainer);

	//display bot's response			
	if (result.output[0].hasOwnProperty('context') &&
		result.output[0].hasOwnProperty('answer') &&
		result.output[0].hasOwnProperty('reference')) {

		let formattedMessage = "";
		// Create bot response for each output item
		for (let item of result.output) {
			// Add answer
			formattedMessage += "<b>Answer:</b><br>" + item.answer + "<br><br>";

			// Add references
			if (Array.isArray(item.reference)) {
				console.log("Reference is an Array");
				formattedMessage += "<b>References:</b><br>";
				for (let i = 0; i < item.reference.length; i++) {
					formattedMessage += (i + 1) + ". " + item.reference[i] + "<br><br>";
				}
			} else {
				console.log("Reference is NOT an Array");
				formattedMessage += "<b>Reference:</b><br>" + item.reference + "<br><br>";
			}

			// Add context
			formattedMessage += "<b>Context:</b><br>" + item.context + "<br><br>";
			formattedMessage += "********************<br>"
			// console.log(formattedMessage);
			console.log(item.context);
		}
		botMessage.innerHTML = formattedMessage;
		botMessageContainer.appendChild(botMessage);
		messageContainer.appendChild(botMessageContainer);
		// console.log(formattedMessage);
	} else {
		console.log("Atleast one of output.context/output.answer/output.reference is NOT a property");
	}
}


// New function to handle login
async function handleLogin(e) {
	e.preventDefault();

	const username = document.getElementById('username').value;
	// console.log ("username = " + username);
	const password = document.getElementById('password').value;
	// console.log ("password = " + password);

	let py_script_path = 'pycodes/validate_user.py';
	let isError = true;
	
	// Validate username and password
	if (!username || !password) {
		Swal.fire({
			icon: 'error',
			title: 'Empty Fields',
			text: 'Username and password must not be empty',
			customClass: {
				container: 'my-swal'
			},
		});
		return true;
	}

	// You can send this data to your server for authentication
	try {
		const response = await fetch('/api/runPythonScript', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				py_script_path: py_script_path,
				username: username,
				password: password
			}),
		});

		let result = await response.json();

		/* 	result is an object with a single property 'result' which is a JSON string
		You need to parse that JSON string to get the actual result object: */
		console.log("Printing result below:");
		console.log(result);
		result = JSON.parse(result.result);
		if (result.code == "error" || result.code == "failure") {
			Swal.fire({
				icon: 'error',
				title: 'Authentication Failed',
				text: result.message,
				customClass: {
					container: 'my-swal'
				},
				allowOutsideClick: false
			});
			isError = true;
		} else if (result.code == "success") {
			Swal.fire({
				icon: 'success',
				title: 'Successfully logged in!',
				text: '',
				customClass: {
					container: 'my-swal'
				},
			});
			// If authentication is successful
			$('#loginModal').hide();
		}
	} catch (error) {
		Swal.fire({
			icon: 'error',
			title: 'Server Error',
			text: 'Could not communicate with the server' + error,
			customClass: {
				container: 'my-swal'
			},
		});
	}
	return true; // Return true to prevent showLoader from closing the Swal modal
}
/* ----##### USER DEFINED FUNCTIONS END #####---- */

/* Handle datatable modal */
$(document).ready(function () {
	// Hide datatable modal on page load
	$('#dataModal').hide();
});

/* Handle login modal */
$(document).ready(function () {
	// Show login modal on page load
	$('#loginModal').show();

	// Attach the login event handler
	// $('#login-button').click(handleLogin);
	// $('#login-button').click(showLoader(handleLogin));
	$('#login-button').on('click', function (e) {
		// showLoader(handleLogin, e);
		showLoader(handleLogin, "Validating login ...", e);
	});


	// Event handler when the 'Enter' key is pressed
	$(document).on('keypress', '#username, #password', function (e) {
		if (e.which === 13) { // 13 is the key code for the 'Enter' key
			// handleLogin(e);
			// showLoader(handleLogin, e);
			showLoader(handleLogin, "Validating login ...", e);
		}
	});
});


/* Show/Hide datatable modal 
(useful when changing parameters like `One paper` or `All paper` after getting the datatable from python code) 
*/
$(document).ready(function () {
	let button_show_datatable = document.getElementById('show_datatable');
	button_show_datatable.addEventListener('click', async () => {
		if ($('#dataModal').is(':hidden')) {
			$('#dataModal').show();
		} else {
			$('#dataModal').hide();
		}
	});
});

/* Handle params modal */
$(document).ready(function () {
	let button_adv_params = document.getElementById('button_adv_params');
	button_adv_params.addEventListener('click', async () => {
		$("#setParametersModal").modal('show');
	});
});


$(document).ready(function () {
	// Hide all tab content initially
	$('#chat_panel').hide();
	$('#pdf_panel').hide();

	// Event listener for tab click
	$('#tab_panels a').click(function (e) {
		e.preventDefault();

		// Get the target panel from the clicked tab's href attribute
		var targetPanel = $(this).attr('href');

		// Hide all panel contents
		$('#chat_panel').hide();
		$('#pdf_panel').hide();

		// Show the targeted panel content
		$(targetPanel).show();

		// Set the clicked tab as active
		$(this).tab('show');
	});
});

$(document).ready(function () {
	// Calculate the heights of the header and footer
	var headerHeight = $("nav.main-header").outerHeight();
	var footerHeight = $("footer.main-footer").outerHeight();
	headerHeight += 100;

	// Include paddings into the calculation
	var paddingTop = parseFloat($('#container_fluid_outer').css('padding-top'));
	// var paddingBot = parseFloat($('#container_fluid_outer').css('padding-bottom'));

	headerHeight += paddingTop - 15;
	// footerHeight += paddingBot;

	// Calculate the appropriate height for #fluid_container1
	var fluidContainerHeight = "calc(100vh - " + headerHeight + "px - " + footerHeight + "px)";
	// console.log("fluidContainerHeight = " + fluidContainerHeight);
	// Apply this height to #fluid_container1
	$("#fluid_container1").css("height", fluidContainerHeight);

	// Calculate the appropriate height for #viewerContainer
	var tabPanelsHeight = $("#tab_panels").outerHeight();
	var pdfPanelsHeight = $("#pdf_panel").outerHeight();
	var pdfPanelNavBarHeight = $("#navigation-bar").outerHeight();


	// Reset the headerHeight to actual header height and then and required offsets
	headerHeight = $("nav.main-header").outerHeight();
	// console.log("headerHeight = " + headerHeight);
	// console.log("tabPanelsHeight = " + tabPanelsHeight);
	// console.log("pdfPanelsHeight = " + pdfPanelsHeight);
	// console.log("pdfPanelNavBarHeight = " + pdfPanelNavBarHeight);
	headerHeight += tabPanelsHeight + pdfPanelsHeight;
	footerHeight += 8;
	var viewerContainerHeight = "calc(100vh - " + headerHeight + "px - " + footerHeight + "px)";
	$("#viewerContainer").css("margin-top", pdfPanelsHeight + 10);
	// $("#pdf_area").css("margin-top", pdfPanelsHeight + 10);
	// var marginTop = parseFloat($('#viewerContainer').css('margin-top'));
	// console.log("marginTop = " + marginTop);

	// Apply this height to #viewerContainer
	// $("#viewerContainer").css("height", viewerContainerHeight);
	$("#pdf_area").css("height", viewerContainerHeight);
});



/* ----##### Set tooltips for params #####---- */
tippy('#tooltip_select_llm', {
	content: 'Select an LLM, GPT-3.5 is much cheaper but less powerful than GPT-4. \
	For complex queries GPT-4 performs significantly better.',
	theme: 'my-tippy-theme'
});

tippy('#tooltip_select_namespace', {
	content: 'Select a relevant namespace. \
	A relevant namespace is the one that contains matching contexts to your query. \
	Selecting irrelevant namespace might result in sub-optimal answer and hence \
	waste tokens/money unnecessarily.',
	theme: 'my-tippy-theme'
});

tippy('#tooltip_search_namespace', {
	content: 'Enter a valid PubMed ID to be searched in the selected namespace.',
	theme: 'my-tippy-theme'
});

tippy('#tooltip_advanced_params', {
	content: 'Advanced params are used to configure the app for more sophistcated search.',
	theme: 'my-tippy-theme'
});

tippy('#tooltip_show_datatable', {
	content: 'Show/Hide datatable modal (obtained after re-ranking chunks by keyword frequencies).',
	theme: 'my-tippy-theme'
});

tippy('#tooltip_top_k', {
	content: 'Enter how many relevant (to query) chunks to be retrieved from vector DB. \
	The minimum is 1 and the maximum depends on the LLM selected. \
	Roughly, for GPT-3.5 it is 7 and for GPT-4 it is 15.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_temp', {
	content: 'Select sampling temperature to use (between 0-2). \
	Higher values like 0.8 will make the output more random, \
	while lower values close to 0 will make it more focused and deterministic. \
	In most of the cases you do not need to change this parameter.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_embedd_model', {
	content: 'Select the embedding model used to convert text into numbers (during namespace generation). \
	We used `biobert` for text embedding and is the only option provided in this version',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_paper_id', {
	content: 'Provide a valid paper ID from the re-ranking table to limit the answer \
	generated using the chunks of that paper only.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_rerank', {
	content: 'Select True, if you want to re-rank chunks \
	based on keyword frequencies. By default, this option is on.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_fix_keyword', {
	content: 'Activate only with `Rerank = True` to use primary keywords \
	to guide re-ranking. See documentation for the algorithm.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_template', {
	content: 'This is field is optional and blank by default. Here you can input further  \
	information that you think might help the model to generate mode precise \
	response to your query. Once set, remember to change/clear this field once you move on to a \
	different query where the current template is not relevant.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_answer_per_paper', {
	content: 'Enable for answers from individual chunks. \
	By default, answers use information from the entire prompt.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_chunks_from_one_paper', {
	content: 'Set to `True` to confine answers to a selected paper, ensuring `paper_id` is set. \
	Off by default.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_select_rows_table', {
	content: 'Enable to choose specific rows from the re-ranking table. \
	Defaults: top 5 for GPT3.5, top 10 for GPT-4.',
	zIndex: 10001,
	theme: 'my-tippy-theme'
});

tippy('#tooltip_keywords', {
	content: 'Enter keywords to be searched in titles/abstracts',
	theme: 'my-tippy-theme'
});

// tippy('#tooltip_template', {
// 	content: 'Enter a template to be prefixed with your prompt',
// 	theme: 'my-tippy-theme'
// });

tippy('#tooltip_date', {
	content: 'Select date ranges',
	theme: 'my-tippy-theme'
});

/* ----##### Dynamically add namespaces #####---- */
window.onload = async function () {
	// Fetch the namespaces from the server-side
	let response = await fetch('/api/getNamespaces');
	let namespaces = await response.json();
	const select = document.getElementById('select_namespace');

	// List of options you want to add
	// let options = Object.keys(namespaces);
	let options = namespaces;

	// Dynamically creating options and appending to select
	options.forEach(optionValue => {
		// console.log("optionValue = " + optionValue);
		let option = document.createElement('option');
		option.value = optionValue;
		option.text = optionValue;
		select.appendChild(option);
	});
}

/* ----##### side window date picker #####---- */
let startDate;
let endDate;

const startInput = document.querySelector("#start-date");
const endInput = document.querySelector("#end-date");

const startPicker = flatpickr(startInput, {
	dateFormat: "Y/m/d",
	onChange: function (selectedDates, dateStr, instance) {
		startDate = selectedDates[0];
		if (endPicker) {
			endPicker.set('minDate', startDate);
		}
	},
});

const endPicker = flatpickr(endInput, {
	dateFormat: "Y/m/d",
	maxDate: new Date(), // set maxDate to current date
	onChange: function (selectedDates, dateStr, instance) {
		endDate = selectedDates[0];
		if (startPicker) {
			startPicker.set('maxDate', endDate);
		}
	},
});


/* ----##### calling python from JS #####---- */
/* Search namespace */
var button_search_namespace = document.getElementById('button_search_namespace');
button_search_namespace.addEventListener('click', async () => {
	var namespace = document.getElementById('select_namespace').value;
	const pmid = document.getElementById('search_namespace').value;
	var py_script_path = 'pycodes/get_pmid_info.py';  // specify python script path here

	// Show intermediate progress bar
	document.getElementById('progress-container').classList.remove('hidden');

	const response = await fetch('/api/runPythonScript', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			py_script_path: py_script_path,
			namespace: namespace,
			pmid: pmid
		}),
	});
	let result = await response.json();
	/* 	result is an object with a single property 'result' which is a JSON string
		You need to parse that JSON string to get the actual result object: */
	// console.log("Printing result below:");
	console.log(result);
	result = JSON.parse(result.result);

	// Hide progress bar when task completes
	document.getElementById('progress-container').classList.add('hidden');

	if (result.code == "failure") {
		Swal.fire({
			icon: 'error',
			title: "Namespace search failed",
			html: result.output,
			allowOutsideClick: false
		});
	} else {
		Swal.fire({
			icon: 'info',
			title: "Namespace search succeeded",
			// html: result.output,
			html: "<div class='align-left'>" + result.output + "</div>", 
			allowOutsideClick: false
		});
	}

});


/* PMC article downloader params */
var button_fetch_articles = document.getElementById('button_fetch_articles');
button_fetch_articles.addEventListener('click', async () => {
	Swal.fire({
		icon: 'info',
		//title: "In this version namespace generation is disabled",
	        title: "Namespace generation is disabled in this version due to resource limitations on the Pinecone server under the free 'Starter' plan. To enable this feature, please set up the app locally by following the documentation titled 'Running WeiseEule as localhost app' in the supplement.",
		allowOutsideClick: false
	});

	// const embedd_model = document.getElementById('select_embedd_model').value;
	// const keywords = document.getElementById('keywords').value;
	// const start_date = document.getElementById('start-date').value;
	// const end_date = document.getElementById('end-date').value;
	// var py_script_path = 'pycodes/fetch_articles.py';  // specify python script path here

	// if (keywords != "" & start_date != "" & end_date != "") {
	// 	// Show intermediate progress bar
	// 	document.getElementById('progress-container').classList.remove('hidden');
	// 	const response = await fetch('/api/runPythonScript', {
	// 		method: 'POST',
	// 		headers: {
	// 			'Content-Type': 'application/json',
	// 		},
	// 		body: JSON.stringify({
	// 			py_script_path: py_script_path,
	// 			embedd_model: embedd_model,
	// 			keywords: keywords,
	// 			start_date: start_date,
	// 			end_date: end_date
	// 		}),
	// 	});
	// 	let result = await response.json();
	// 	console.log(result);
	// 	// Now, result is an object with a single property 'result' which is a JSON string
	// 	// You need to parse that JSON string to get the actual result object:
	// 	result = JSON.parse(result.result);
	// 	console.log(result.message);

	// 	// Hide progress bar when task completes
	// 	document.getElementById('progress-container').classList.add('hidden');

	// 	if (result.code == "exit" || result.code == "failure") {
	// 		Swal.fire({
	// 			icon: 'warning',
	// 			title: "Exit Downloader",
	// 			text: result.message,
	// 			allowOutsideClick: false
	// 		});
	// 	} else if (result.code == "success") {
	// 		Swal.fire({
	// 			icon: 'success',
	// 			title: result.code.toUpperCase(),
	// 			text: result.message,
	// 			allowOutsideClick: false
	// 		});
	// 	}
	// 	// console.log(result);	
	// } else {
	// 	Swal.fire({
	// 		icon: 'error',
	// 		title: 'Empty params',
	// 		text: 'Values of keywords, start_date & end_date must not be empty',
	// 		allowOutsideClick: false
	// 	});
	// }
});

/* ----##### Main window chatbox #####---- */
async function showLoader(asyncFunctionToRun, title = "Getting the answer, wait ...", ...params) {
	const temp = parseFloat(document.getElementById('set_temp').value);
	if (temp < 0 || temp > 2) {
		console.log("showLoader() should return now");
		Swal.fire({
			icon: 'error',
			title: 'Invalid params',
			text: 'temperature parameter must be between 0 and 2',
			allowOutsideClick: false
		});
		return;
	}

	Swal.fire({
		// title: 'Getting the answer, wait ...',
		title: title,
		// background: '#d6d7d8',
		background: 'rgb(212, 209, 200)',
		didOpen: () => {
			Swal.showLoading()
		},
		didClose: () => {
			Swal.hideLoading()
		},
		allowOutsideClick: false,
		allowEscapeKey: false,
		allowEnterKey: false
	});

	let error = await asyncFunctionToRun(...params);
	if (!error) {
		Swal.close();
	}
}

document.getElementById('send-button').addEventListener('click', async function (event) {
	console.log("Calling showLoader(sendInput) from addEventListener('click')");
	showLoader(sendInput); // sendInput is an async function defined later
});

document.getElementById('user-input').addEventListener('keydown', async function (event) {
	if (event.key === 'Enter') {
		event.preventDefault();
		console.log("Calling showLoader(sendInput) from addEventListener('keydown')");
		showLoader(sendInput); // sendInput is an async function defined later
	}
});

async function sendInput() {
	// var llm = document.getElementById('select_llm').value;
	// const temp = parseFloat(document.getElementById('set_temp').value);
	// const namespace = document.getElementById('select_namespace').value;
	// const template = document.getElementById('template').value;
	// const top_k = document.getElementById('set_top_k').value;
	// const embedd_model = document.getElementById('select_embedd_model').value;
	// const paper_id = document.getElementById('set_paper_id').value;
	// // const search_keywords = document.getElementById('search_keywords').value.toLowerCase();
	// // const primary_keywords = document.getElementById('primary_keywords').value.toLowerCase();
	// const fix_keyword = document.getElementById('select_fix_keyword').value;
	// const answer_per_paper = document.getElementById('select_answer_per_paper').value;
	// const chunks_from_one_paper = document.getElementById('select_chunks_from_one_paper').value;
	// const rerank = document.getElementById('select_rerank').value;
	// const advanced_use = document.getElementById('select_rows_table').value;

	var llm = document.getElementById('select_llm').value;
	var temp = parseFloat(document.getElementById('set_temp').value);
	var namespace = document.getElementById('select_namespace').value;
	var template = document.getElementById('template').value;
	var top_k = document.getElementById('set_top_k').value;
	var embedd_model = document.getElementById('select_embedd_model').value;
	var paper_id = document.getElementById('set_paper_id').value;
	// var search_keywords = document.getElementById('search_keywords').value.toLowerCase();
	// var primary_keywords = document.getElementById('primary_keywords').value.toLowerCase();
	var fix_keyword = document.getElementById('select_fix_keyword').value;
	var answer_per_paper = document.getElementById('select_answer_per_paper').value;
	var chunks_from_one_paper = document.getElementById('select_chunks_from_one_paper').value;
	var rerank = document.getElementById('select_rerank').value;
	var advanced_use = document.getElementById('select_rows_table').value;
	console.log("advanced_use = " + advanced_use);


	var py_script_path = 'pycodes/get_answer.py';
	let isError = false;

	if (temp < 0 || temp > 2) {
		console.log("sendInput() should return now");
		// return { icon: 'error', title: 'Invalid params', text: 'temperature parameter must be between 0 and 2' };
	} else {
		let inputField = document.getElementById('user-input');
		let userInput = inputField.value;

		/* Here you should process the user input and generate a response */
		console.log("Calling runPythonScript API from sendInput() " + ++count + " times!");
		const response = await fetch('/api/runPythonScript', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				py_script_path: py_script_path,
				llm: llm,
				temp: temp,
				namespace: namespace,
				query: userInput.toLowerCase(),
				template: template,
				// search_keywords: search_keywords,
				// primary_keywords: primary_keywords,
				embedd_model: embedd_model,
				paper_id: paper_id,
				answer_per_paper: answer_per_paper,
				chunks_from_one_paper: chunks_from_one_paper,
				fix_keyword: fix_keyword,
				rerank: rerank,
				top_k: top_k
			}),
		});
		let result = await response.json();

		/* 	result is an object with a single property 'result' which is a JSON string
		You need to parse that JSON string to get the actual result object: */
		// console.log("Printing result below:");
		console.log(result);
		result = JSON.parse(result.result);
		// console.log("Printing result.code below:");
		// console.log(result.code);
		if (result.code == "error" || result.code == "failure") {
			Swal.fire({
				icon: 'error',
				title: "Error in chat completion",
				text: result.output,
				allowOutsideClick: false
			});
			isError = true;
		} else if (result.code == "success" && rerank === 'True') {
			/* write logic to display the df as datatable */
			//show datatable modal
			var modal = document.getElementById("dataModal");
			// var span = document.getElementsByClassName("close")[0];
			var span = document.getElementById("dataModal-close");
			span.onclick = function () {
				modal.style.display = "none";
				$('#dataTable').off('click', 'tbody tr');  // Optionally, you may try unbinding the click event here
			}

			/* Check and Destroy Previous DataTable Instances */
			if ($.fn.DataTable.isDataTable('#dataTable')) {
				$('#dataTable').DataTable().destroy();
				$('#dataTable tbody').empty();
				$('#dataTableHeaders').empty(); // Clear the previously injected headers
			}

			// Get keys from the first item of result.output
			var keys = Object.keys(result.output[0]);

			// Create an array of column definitions using the keys
			var columnsDef = keys.map(key => {
				return { data: key };
			});

			// Dynamically generate table headers and append to the table
			keys.forEach(key => {
				$('#dataTableHeaders').append('<th>' + key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.slice(1) + '</th>');
			});

			var table = $('#dataTable').DataTable({
				data: result.output,
				columns: columnsDef,
				order: [],
				responsive: true
			});

			/* Show the table upon successful python execution */
			$('#dataModal').show();

			// $('#dataTable tbody tr').off('click').on('click', function () {
			// 	let select_rows_setting = $('#select_rows_table').val();
			// 	if (select_rows_setting === 'True') {
			// 		console.log('Row clicked!');
			// 		$(this).toggleClass('selected');
			// 	} else {
			// 		console.log('select_rows_setting = ' + select_rows_setting);
			// 	}
			// });

			$('#dataTable tbody').off('click').on('click', 'tr', function () {
				let select_rows_setting = $('#select_rows_table').val();
				if (select_rows_setting === 'True') {
					console.log('Row clicked!');
					$(this).toggleClass('selected');
				} else {
					console.log('select_rows_setting = ' + select_rows_setting);
				}
			});

			$('#getIndicesBtn').off('click').on('click', function () {
				var selectedRows = table.rows('.selected').data();
				llm = document.getElementById('select_llm').value;

				if (selectedRows.length === 0) {
					// console.log('No rows selected!');
					// return;
					/* Default case for normal user */
					// const defaultSelectCount = llm == 'gpt-3.5-turbo' ? 5 : (llm == 'gpt-4' ? 10 : 0);
					const defaultSelectCount = (() => {
						switch (llm) {
							case 'gpt-3.5-turbo':
								return 5;
							case 'gpt-3.5-turbo-1106':
								return 10;
							case 'gpt-4-1106-preview':
								return 10;
							case 'gpt-4':
								return 8;
							// Add more cases as needed
							default:
								return 5;
						}
					})();
					console.log("llm = " + llm);
					console.log("defaultSelectCount = " + defaultSelectCount);

					// Function to extract 'vector_id' from the first N rows, where N = defaultSelectCount
					var getDefaultVectorIds = () => {
						return result.output.slice(0, defaultSelectCount).map(row => row['vector_id']);
					};
					var vectorIds = getDefaultVectorIds();
					console.log("Selected vector ids: ", vectorIds);
				} else {
					console.log("Selected Indices: ", table.rows('.selected').indexes().toArray());
					var vectorIds = [];
					$.each(selectedRows, function (index, value) {
						vectorIds.push(value['vector_id']);
					});
					console.log("Selected vector ids: ", vectorIds);
				}

				console.log("Calling showLoader(sendVectorIds) from $('#getIndicesBtn').click");
				/* Call an asynchronous function and pass vectorIds */
				showLoader(sendVectorIds,
					"Getting the answer, wait ...",
					vectorIds,
					namespace,
					template,
					userInput);
			});
		}
		else {
			/* display user message on console */
			await displayChatResponse(result, userInput);
		}

		// Clear the input field
		inputField.value = '';
	}
	return isError;
}

/* ----##### Main window PDFviewer #####---- */
let prev_valid_page_num = 1;

document.addEventListener("DOMContentLoaded", function () {
	pdfjsLib.GlobalWorkerOptions.workerSrc = '/js/pdf.worker.js';


	var eventBus = new pdfjsViewer.EventBus();
	var pdfViewer = new pdfjsViewer.PDFViewer({
		container: document.getElementById('viewerContainer'),
		viewer: document.getElementById('pdfViewer'),
		eventBus: eventBus,
		textLayerMode: 2,
	});


	// document.getElementById('file-input').addEventListener('change', function (event) {
	// 	let file = event.target.files[0];
	// 	if (file.type !== 'application/pdf') {
	// 		console.error(file.name, 'is not a .pdf file');
	// 		return;
	// 	}

	// 	let fileReader = new FileReader();
	// 	fileReader.onload = function (event) {
	// 		let typedArray = new Uint8Array(this.result);
	// 		pdfjsLib.getDocument(typedArray).promise.then((pdf) => {
	// 			pdfViewer.setDocument(pdf);

	// 			// set max value of input field
	// 			document.getElementById('go-to-page').max = pdf.numPages;
	// 			document.getElementById('go-to-page').value = 1;
	// 		});
	// 	};
	// 	fileReader.readAsArrayBuffer(file);
	// });

	// Temporarily blocking the file upload
	document.getElementById('file-upload-label').addEventListener('click', function (e) {
		e.preventDefault();  // Prevent the default behavior of the label
		Swal.fire({
			icon: 'info',
			//title: "In this version pdf upload is disabled",
		        title: "PDF upload is disabled in this version due to limited resources on the cloud server hosting the app. To enable this feature, please set up the app locally by following the documentation titled 'Running WeiseEule as localhost app' in the supplement.",
			allowOutsideClick: false
		});
	});

	document.getElementById('zoom-in').addEventListener('click', function () {
		pdfViewer.currentScaleValue = pdfViewer.currentScale + 0.5;
	});

	document.getElementById('zoom-out').addEventListener('click', function () {
		pdfViewer.currentScaleValue = pdfViewer.currentScale - 0.5;
	});

	document.getElementById('prev-page').addEventListener('click', function () {
		console.log("next button is clicked " + ++count_prev + " times!")
		if (pdfViewer.currentPageNumber > 1) {
			pdfViewer.currentPageNumber--;
			prev_valid_page_num = pdfViewer.currentPageNumber;
			document.getElementById('go-to-page').value = prev_valid_page_num;
		}
	});

	document.getElementById('next-page').addEventListener('click', function () {
		console.log("next button is clicked " + ++count_next + " times!");
		if (pdfViewer.currentPageNumber < pdfViewer.pagesCount) {
			pdfViewer.currentPageNumber++;
			prev_valid_page_num = pdfViewer.currentPageNumber;
			document.getElementById('go-to-page').value = prev_valid_page_num;
		}
	});

	document.getElementById('go-to-page').addEventListener('change', function () {
		// let pageNumber = this.value;
		let pageNumber = parseInt(this.value);
		if (pageNumber > 0 && pageNumber <= pdfViewer.pagesCount) {
			pdfViewer.currentPageNumber = pageNumber;
			prev_valid_page_num = pageNumber;
		} else {
			document.getElementById('go-to-page').value = prev_valid_page_num;
		}
	});



	let actionCard = document.getElementById('action-card');
	let copyBtn = document.getElementById('copy-button');
	let summarizeBtn = document.getElementById('summarize-button');

	document.addEventListener('mouseup', (event) => {
		let selection = window.getSelection();
		if (selection.toString().length > 0) {
			// Show action card
			actionCard.style.display = 'flex';

			// Position action card at cursor location
			actionCard.style.top = `${event.pageY}px`;
			actionCard.style.left = `${event.pageX}px`;
		} else {
			// Hide action card
			actionCard.style.display = 'none';
		}
	});

	copyBtn.addEventListener('click', async () => {
		// Copy selected text to clipboard
		try {
			await navigator.clipboard.writeText(window.getSelection().toString());
		} catch (err) {
			console.error('Failed to copy text: ', err);
		}

		// Hide action card
		actionCard.style.display = 'none';
	});

	// Update the event listener for the 'Summarize' button to hide the summary card before showing it with new content.
	summarizeBtn.addEventListener('click', async () => {
		Swal.fire({
			title: 'Getting the summary, wait ...',
			// background: '#d6d7d8',
			background: 'rgb(212, 209, 200)',
			didOpen: () => {
				Swal.showLoading()
			},
			didClose: () => {
				Swal.hideLoading()
			},
			allowOutsideClick: false,
			allowEscapeKey: false,
			allowEnterKey: false
		});
		let selection = window.getSelection();
		const text = selection.toString();
		var llm = document.getElementById('select_llm').value;
		var py_script_path = 'pycodes/get_summary.py';

		// Hide action card
		actionCard.style.display = 'none';
		// Hide summary card (in case it's already shown)
		let summaryCard = document.getElementById('summary-card');
		summaryCard.style.display = 'none';
		// Update summary card content
		const response = await fetch('/api/runPythonScript', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				py_script_path: py_script_path,
				llm: llm,
				text: text
			}),
		});
		let result = await response.json();
		Swal.close();

		/* 	result is an object with a single property 'result' which is a JSON string
		You need to parse that JSON string to get the actual result object: */
		console.log(result);
		result = JSON.parse(result.result);
		if (result.code == "error") {
			Swal.fire({
				icon: 'error',
				title: "Error in chat completion",
				text: result.message,
				allowOutsideClick: false
			});
			// isError = true;
		} else {
			document.getElementById('summary-content').textContent = result.message;
			// Show summary card
			setTimeout(() => {
				summaryCard.style.display = 'block';
			}, 0);
		}
	});

	// Add an event listener to the close button to hide the summary card when clicked.
	document.getElementById('close-summary').addEventListener('click', () => {
		document.getElementById('summary-card').style.display = 'none';
	});
});
