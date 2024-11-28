"use strict";
/* ----##### Import packages BELOW #####---- */
// import { createApp } from 'vue'
import tippy from 'tippy.js'
import 'tippy.js/dist/tippy.css'
import Swal from 'sweetalert2'
import flatpickr from "flatpickr";
import "flatpickr/dist/flatpickr.min.css";
// let wsUrl = "ws://localhost:8000/stream_answer";
let wsUrl = "ws://localhost:8000/ws";
// let wsUrl = "wss://www.weiseeule.info/ws";
let ws; // Declare WebSocket variable outside to manage its state

let count_next = 0;
let count_prev = 0;
let count = 0;
let tableCounter = 0; // Counter to ensure unique table IDs
/* ----##### USER DEFINED FUNCTIONS START #####---- */
function appendDataFrame(df) {
	const parsedData = JSON.parse(df);
	var modal = document.getElementById("dataModal");
	var span = document.getElementById("dataModal-close");

	span.onclick = function () {
		modal.style.display = "none";
		$('#dataTable').off('click', 'tbody tr');
	}

	// Check and Destroy Previous DataTable Instances
	if ($.fn.DataTable.isDataTable('#dataTable')) {
		$('#dataTable').DataTable().destroy();
		$('#dataTable tbody').empty();
		$('#dataTableHeaders').empty();
	}

	// Use parsedData.columns directly since it's already an array
	var columnsDef = parsedData.columns.map(col => ({ title: col }));

	// Dynamically generate table headers
	parsedData.columns.forEach(col => {
		$('#dataTableHeaders').append('<th>' + col.replace(/_/g, ' ').charAt(0).toUpperCase() + col.slice(1) + '</th>');
	});

	// Initialize DataTable
	var table = $('#dataTable').DataTable({
		data: parsedData.data,
		columns: columnsDef,
		order: [],
		responsive: true
	});

	/* Show the modal upon successful DataFrame reception */
	// modal.style.display = "block";
	// $('#dataModal').show();

	/* In the updated code hide the DataFrame */
	$('#dataModal').hide();
}
/* #################################################### */

/* Append search DataFrame as DataTable separately for each question */
function appendSearchDataFrame(df, container, query) {
	const parsedData = JSON.parse(df);

	// Increment the table counter to create a unique ID
	tableCounter++;
	const tableId = `dataTableSearch${tableCounter}`;

	// Create table structure
	const table = document.createElement('table');
	table.id = tableId;
	table.className = 'display'; // Only use display class if necessary

	const thead = document.createElement('thead');
	const tr = document.createElement('tr');

	// Add a header for the checkbox column
	const thCheckbox = document.createElement('th');
	thCheckbox.innerHTML = '';
	tr.appendChild(thCheckbox);

	parsedData.columns.forEach(col => {
		const th = document.createElement('th');
		th.textContent = col.replace(/_/g, ' ').charAt(0).toUpperCase() + col.slice(1);
		tr.appendChild(th);
	});


	thead.appendChild(tr);
	table.appendChild(thead);

	const tbody = document.createElement('tbody');
	table.appendChild(tbody);

	container.appendChild(table);

	// Map data to match DataTables format and add checkboxes
	const data = parsedData.data.map(row => {
		return ['', ...row];
	});

	// Initialize DataTable with unique ID
	$(`#${tableId}`).DataTable({
		data: data,
		columns: [
			{ title: '', orderable: false }, // Checkbox column
			...parsedData.columns.map(col => ({ title: col }))
		],
		columnDefs: [
			{
				orderable: false,
				render: DataTable.render.select(),
				targets: 0
			}
		],
		order: [],
		fixedColumns: {
			start: 2
		},
		select: {
			style: 'multi',
			selector: 'td:first-child'
		},
		responsive: true,
		layout: {
			topStart: {
				buttons: [
					{
						text: 'Summarize',
						action: function () {
							// Create a div to display the PMIDs
							const summaryDiv = document.createElement('div');
							summaryDiv.id = `summaryDiv${tableCounter}`;
							container.appendChild(summaryDiv);

							var my_table = $(`#${tableId}`).DataTable();
							let count = my_table.rows({ selected: true }).count();
							let selected_data = my_table.rows({ selected: true }).data();
							console.log(count + ' row(s) selected');

							// Initialize the new array to store the extracted numbers
							let extractedNumbers = [];

							// Iterate through each element in the selected_data array
							for (var i = 0; i < selected_data.length; i++) {
								console.log('selected_data[' + i + ']:- ', selected_data[i]);
								console.log('selected_data[' + i + '][1]:- ', selected_data[i][1]);
								let parts = selected_data[i][1].split('>');
								if (parts.length > 1) {
									let numberPart = parts[1].split('<')[0];
									console.log('numberPart:-', numberPart);
									extractedNumbers.push(numberPart);
								}
							}
							// Output the new array
							console.log('extractedNumbers:-', extractedNumbers); // Output: ["34000094", "34000005"]

							// Send the selected PMIDs to the backend for processing
							sendPMIDsToBackend(extractedNumbers.join(','), `summaryDiv${tableCounter}`, query);
						}
					}
				]
			}
		}
	});

	// Add dynamic CSS rules
	addDynamicTableStyles(tableId);
}


// Function to send PMIDs to python backend based on checkbox selection
function sendPMIDsToBackend(pmids, summaryDivId, query) {
	var llm = document.getElementById('select_llm').value;
	ws = new WebSocket(wsUrl + "/summarize_abstracts");

	const params = {
		llm: llm,
		pmids: pmids,
		query: query
	}

	ws.onopen = () => {
		console.log("WebSocket connection opened.");
		ws.send(JSON.stringify(params));

		// Clear the previous summary content below the "Summary:" heading
		const summaryDiv = document.getElementById(summaryDivId);
		summaryDiv.innerHTML = "<strong>Summary:</strong><br>";
	};

	ws.onmessage = (event) => {
		const data = JSON.parse(event.data);
		const { summary, error, end_summary } = data;

		if (error) {
			console.error(error);
			Swal.fire({
				icon: 'error',
				title: "Error in summarization",
				text: error,
				allowOutsideClick: false
			});
			return;
		}

		if (summary) {
			document.getElementById(summaryDivId).innerHTML += summary;
		}

		if (end_summary) {
			console.log("End of Summary");
			ws.close();
		}
	};

	ws.onclose = () => {
		console.log("WebSocket connection closed.");
	};

	ws.onerror = (error) => {
		console.error("WebSocket error:", error);
		ws.close();
	};
}

/* #################################################### */

function addDynamicTableStyles(tableId) {
	const style = document.createElement('style');
	// style.type = 'text/css';
	style.innerHTML = `
        #${tableId} {
            width: 100% !important;
            border-collapse: collapse !important;
        }
        #${tableId} th {
            background-color: #7491c4;
            color: black;
        }
        #${tableId} tbody tr:nth-child(odd) {
            background-color: #c2b4d2;
        }
        #${tableId} tbody tr:nth-child(even) {
            background-color: rgb(222, 206, 235);
        }
        #${tableId} tbody tr:hover {
            background-color: #9bbffc;
        }
        #${tableId} th, #${tableId} td {
            padding: 8px;
            // border: 1px solid #005eff;
			border: 1px solid #A0AFB7;
            text-align: left;
        }
    `;
	document.getElementsByTagName('head')[0].appendChild(style);
}
/* #################################################### */

function setupChatElements(ws, params) {
	// Prepare message containers for chat
	const elements = createMessageContainerElements('chat-area');
	ws.chatArea = elements.chatArea;
	ws.messageContainer = elements.messageContainer;

	const userElements = createUserMessageElements();
	ws.userMessageContainer = userElements.userMessageContainer;
	ws.userMessage = userElements.userMessage;

	const botElements = createBotMessageElements();
	ws.botMessageContainer = botElements.botMessageContainer;
	ws.botMessage = botElements.botMessage;

	// Display user's query
	ws.userMessage.textContent = params.query;
	ws.userMessageContainer.appendChild(ws.userMessage);
	ws.messageContainer.appendChild(ws.userMessageContainer);

	let formattedMessage = "<b>Answer:</b><br>";
	ws.botMessage.innerHTML += formattedMessage;
	ws.botMessageContainer.appendChild(ws.botMessage);
	ws.messageContainer.appendChild(ws.botMessageContainer);
}

function setupSearchElements(ws, params) {
	// Prepare message containers for search
	const elements = createMessageContainerElements('search-area');
	ws.searchArea = elements.chatArea;
	ws.messageContainer = elements.messageContainer;

	const userElements = createUserMessageElements();
	ws.userMessageContainer = userElements.userMessageContainer;
	ws.userMessage = userElements.userMessage;

	const botElements = createBotMessageElements();
	ws.botMessageContainer = botElements.botMessageContainer;
	ws.botMessage = botElements.botMessage;

	// Display user's query
	ws.userMessage.textContent = params.query;
	ws.userMessageContainer.appendChild(ws.userMessage);
	ws.messageContainer.appendChild(ws.userMessageContainer);

	let formattedMessage = "<b>Top 10 Relevant PMIDs:</b><br><br>";
	ws.botMessage.innerHTML += formattedMessage;
	ws.botMessageContainer.appendChild(ws.botMessage);
	ws.messageContainer.appendChild(ws.botMessageContainer);
}
/* #################################################### */

function handleSearchPubmedMessage(event, ws, params) {
	/* Stop loader on first message */
	Swal.close();

	/* Scroll search area top to make strem visble continuously */
	ws.searchArea.scrollTop = ws.searchArea.scrollHeight;

	const data = JSON.parse(event.data);
	const { content, citation, last_content, error, df } = data;

	if (error) {
		console.error(error);
		Swal.fire({
			icon: 'error',
			title: "Error in PubMed Search",
			text: error,
			allowOutsideClick: false
		});
		ws.close();
		// return;
	}

	let formattedMessage = "";

	if (citation) {
		formattedMessage = "<br><br><b>Citation:</b><br>" + citation;
		ws.botMessage.innerHTML += formattedMessage;
	}

	if (content) {
		ws.botMessage.innerHTML += content;
	}

	if (df) {
		console.log('Received dataframe:');
		appendSearchDataFrame(df, ws.botMessage, params.query);
	}

	if (last_content) {
		ws.close();
	}
}

/* #################################################### */

/* JavaScript function to toggle the context visibility */
function toggleContext(contextId) {
	const contextDiv = document.getElementById(contextId);
	const toggleSymbol = contextDiv.previousElementSibling;

	if (contextDiv.style.display === "none") {
		contextDiv.style.display = "block";
		toggleSymbol.textContent = "[-]";
	} else {
		contextDiv.style.display = "none";
		toggleSymbol.textContent = "[+]";
	}
}


function handleStreamAnswerMessage(event, ws, params) {
	/* Stop loader on first message */
	Swal.close();

	/* Scroll chat area top to make strem visble continuously */
	ws.chatArea.scrollTop = ws.chatArea.scrollHeight;

	const data = JSON.parse(event.data);
	const { content, citation, context, last_context, error, df } = data;

	if (error) {
		console.error(error);
		Swal.fire({
			icon: 'error',
			title: "Error in chat completion",
			text: error,
			allowOutsideClick: false
		});
		ws.close();
		return;
	}

	let formattedMessage = "";

	if (citation) {
		formattedMessage = "<br><br><b>Citation:</b><br>" + citation;
		ws.botMessage.innerHTML += formattedMessage;
	}

	/* Working */
	if (context) {
		// Create a toggle container for the context
		const contextId = `context-${Date.now()}`; // Unique ID for each context

		// Append HTML for context with toggle functionality
		formattedMessage = `
            <br><br>
            <b>Context:</b>
            <span class="toggle-symbol" data-toggle-id="${contextId}">[+]</span>
            <div id="${contextId}" class="context-content" style="display: none;">${context}</div>
        `;
		ws.botMessage.innerHTML += formattedMessage;

		// Add event listener to toggle the visibility
		const toggleSymbol = document.querySelector(`.toggle-symbol[data-toggle-id="${contextId}"]`);
		toggleSymbol.addEventListener('click', function () {
			toggleContext(contextId, toggleSymbol);
		});

		if (params.answer_per_paper === 'True') {
			formattedMessage = "<br><b>Answer:</b><br>";
			ws.botMessage.innerHTML += formattedMessage;
		} else if (params.rerank == 'False') {
			ws.close();
		}
	}

	if (last_context) {
		const lastContextId = `last-context-${Date.now()}`;
		formattedMessage = `
            <br><br>
            <b>Context:</b>
            <span class="toggle-symbol" data-toggle-id="${lastContextId}">[+]</span>
            <div id="${lastContextId}" class="context-content" style="display: none;">${last_context}</div>
        `;
		// formattedMessage = "<br><br><b>Context:</b><br>" + last_context;
		ws.botMessage.innerHTML += formattedMessage;

		// Add event listener to toggle the visibility
		const toggleSymbol = document.querySelector(`.toggle-symbol[data-toggle-id="${lastContextId}"]`);
		toggleSymbol.addEventListener('click', function () {
			toggleContext(lastContextId, toggleSymbol);
		});

		if (params.rerank == 'False') {
			ws.close();
		}
	}

	if (content) {
		ws.botMessage.innerHTML += content;
	}

	if (df) {
		appendDataFrame(df);
		ws.close();
	}
}
/* #################################################### */

function openWebSocket(end_point, params, setupElements, messageHandler) {
	ws = new WebSocket(wsUrl + end_point);

	ws.onopen = () => {
		console.log("WebSocket connection opened.");

		// Send params only when WebSocket connection is open
		ws.send(JSON.stringify(params));
		console.log('Params sent to WebSocket: ', JSON.stringify(params));

		setupElements(ws, params);
	};

	ws.onmessage = (event) => {
		messageHandler(event, ws, params);
	};

	ws.onclose = () => {
		console.log("WebSocket connection closed.");
	};

	ws.onerror = (error) => {
		console.error("WebSocket error:", error);
		ws.close();
	};
}

function serverError(error_message) {
	Swal.fire({
		icon: 'error',
		title: 'Server Error',
		text: `Could not communicate with the server. ${error_message}`,
		customClass: {
			container: 'my-swal'
		},
	});
}

function createMessageContainerElements(id_div) {
	let chatArea = document.getElementById(id_div);
	let messageContainer = document.getElementById(`${id_div}-message-container`);
	if (!messageContainer) {
		messageContainer = document.createElement('div');
		messageContainer.id = `${id_div}-message-container`;
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


// New function to handle login
async function handleLogin(e) {
	e.preventDefault();

	const username = document.getElementById('username').value;
	const password = document.getElementById('password').value;

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
		return true; // Keep the Swal open
	}

	// You can send this data to your server for authentication
	try {
		const response = await fetch('/api/validate_user/', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ username, password }),
		});

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		// Now the response is already a JSON object, so no need for JSON.parse
		const result = await response.json();

		console.log("Printing result below:");
		console.log(result);

		// Handle the response based on the result
		if (result.code === "success") {
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
		} else {
			Swal.fire({
				icon: 'error',
				title: 'Authentication Failed',
				text: result.message,
				customClass: {
					container: 'my-swal'
				},
				allowOutsideClick: false
			});
		}
	} catch (error) {
		serverError(error.message);
	}
	return true; // Keep the Swal open in case of an error
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
	// $('#chat_panel').hide();
	$('#search_panel').hide();
	$('#pdf_panel').hide();

	// Event listener for tab click
	$('#tab_panels a').click(function (e) {
		e.preventDefault();

		// Get the target panel from the clicked tab's href attribute
		var targetPanel = $(this).attr('href');

		// Hide all panel contents
		$('#chat_panel').hide();
		$('#search_panel').hide();
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
	$("#fluid_container2").css("height", fluidContainerHeight);
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

// tippy('#tooltip_review_mode', {
// 	content: 'Search PubMed for relevant hits. \
// 	Set this to `True` when you want to find article relevant to you keyword or query. \
// 	Use `Search` panel to type your keyword when using this feature.',
// 	theme: 'my-tippy-theme'
// });

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
	const namespace = document.getElementById('select_namespace').value;
	const pmid = document.getElementById('search_namespace').value;
	console.log("Selected namespace = " + namespace);
	console.log("Entered PMID = " + pmid);

	const params = {
		namespace: namespace,
		pmid: pmid
	}

	// Show intermediate progress bar
	document.getElementById('progress-container').classList.remove('hidden');
	const end_point = "/search_PMID_in_namespace";
	ws = new WebSocket(wsUrl + end_point);

	ws.onopen = () => {
		console.log("WebSocket connection opened.");
		ws.send(JSON.stringify(params));
	};

	ws.onmessage = (event) => {
		console.log('event.data = ', event.data);
		const result = JSON.parse(event.data);
		document.getElementById('progress-container').classList.add('hidden');

		// Handle the response based on the result
		if (result.code === "failure") {
			console.error(result.code);
			Swal.fire({
				icon: 'error',
				title: 'Namespace search failed',
				html: result.msg,
				customClass: {
					container: 'my-swal'
				},
				allowOutsideClick: false
			});
			return;
		}

		if (result.code === "not found") {
			console.error(result.code);
			Swal.fire({
				icon: 'warning',
				title: 'PMID not found',
				html: result.msg,
				customClass: {
					container: 'my-swal'
				},
				allowOutsideClick: false
			});
			return;
		}

		if (result.code === "success") {
			console.error(result.code);
			Swal.fire({
				icon: 'success',
				title: 'Search succeeded',
				html: result.msg,
				customClass: {
					container: 'my-swal'
				},
				allowOutsideClick: false
			});
			return;
		}
	};

	ws.onerror = (error) => {
		console.error("WebSocket error:", error);
		ws.close();
	};

	ws.onclose = () => {
		console.log("WebSocket connection closed.");
	};
});


/* PMC article downloader params (WebSocket version) */
var button_fetch_articles = document.getElementById('button_fetch_articles');
button_fetch_articles.addEventListener('click', async () => {
	const embedd_model = document.getElementById('select_embedd_model').value;
	const keywords = document.getElementById('keywords').value;
	const start_date = document.getElementById('start-date').value;
	const end_date = document.getElementById('end-date').value;

	const params = {
		embedd_model: embedd_model,
		keywords: keywords,
		start_date: start_date,
		end_date: end_date
	}

	if (keywords != "" & start_date != "" & end_date != "") {
		// Show intermediate progress bar
		document.getElementById('progress-container2').classList.remove('hidden');
		const end_point = "/fetch_articles";
		ws = new WebSocket(wsUrl + end_point);

		ws.onopen = () => {
			console.log("WebSocket connection opened.");
			ws.send(JSON.stringify(params));
		};

		ws.onmessage = (event) => {
			console.log('event.data = ', event.data);
			const result = JSON.parse(event.data);
			document.getElementById('progress-container2').classList.add('hidden');

			// Handle the response based on the result
			if (result.code === "exit") {
				console.warn(result.code);
				Swal.fire({
					icon: 'warning',
					title: 'Article fetch exited',
					html: result.message,
					customClass: {
						container: 'my-swal'
					},
					allowOutsideClick: false
				});
				ws.close(); // Close explicitly
				return;
			} else if (result.code === "failure") {
				console.error(result.code);
				Swal.fire({
					icon: 'error',
					title: 'Article fetch failed',
					html: result.message,
					customClass: {
						container: 'my-swal'
					},
					allowOutsideClick: false
				});
				ws.close(); // Close explicitly
				return;
			} else if (result.code === "success") {
				console.log(result.code);
				Swal.fire({
					icon: 'success',
					title: 'Articles fetched',
					html: result.message,
					customClass: {
						container: 'my-swal'
					},
					allowOutsideClick: false
				});
				ws.close(); // Close explicitly
				return;
			}
		};

		ws.onerror = (error) => {
			console.error("WebSocket error:", error);
			ws.close();
		};

		ws.onclose = () => {
			console.log("WebSocket connection closed.");
		};
	}


	// Swal.fire({
	// 	icon: 'info',
	// 	// title: "In this version namespace generation is disabled",
	// 	title: "Namespace generation is disabled in this version due to resource limitations on the Pinecone server under the free 'Starter' plan. To enable this feature, please set up the app locally by following the instructions in the supplementary file (Additional file 2).",
	// 	allowOutsideClick: false
	// });

	/*------------ DO NOT DELETE ------------*/
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
	/*------------ DO NOT DELETE ------------*/
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
	// if (!error) {
	// 	Swal.close();
	// }
}

/* Call helper function for chat */
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

/* Call helper function for PubMed search */
document.getElementById('search-pubmed-button').addEventListener('click', async function (event) {
	console.log("Calling showLoader(sendInputSearch) from addEventListener('click')");
	showLoader(sendInputSearch); // sendInputSearch is an async function defined later
});

document.getElementById('search-input').addEventListener('keydown', async function (event) {
	if (event.key === 'Enter') {
		event.preventDefault();
		console.log("Calling showLoader(sendInputSearch) from addEventListener('keydown')");
		showLoader(sendInputSearch); // sendInputSearch is an async function defined later
	}
});


async function sendInputSearch() {
	// var search_pubmed = document.getElementById('review_mode').value;
	let isError = false;

	/* Extract the query and clear input field */
	let inputField = document.getElementById('search-input');
	let userInput = inputField.value;
	inputField.value = '';

	/* Here you should process the user input and generate a response */
	console.log("Calling fastAPI get_answer endpoint from sendInputSearch() " + ++count + " times!");
	var query = userInput.toLowerCase();
	try {
		const end_point = "/search_pubmed"
		const params = {
			// search_pubmed: search_pubmed,
			query: query
		};
		// console.log('Before openWebSocket()');
		console.log('Before calling openWebSocket() params.query = ' + params.query);
		// openWebSocket(end_point, params);
		// openWebSocket(end_point, params, handleSearchPubmedMessage);
		openWebSocket(end_point, params, setupSearchElements, handleSearchPubmedMessage);
		console.log('After openWebSocket()');
	} catch (error) {
		// // Clear the input field
		// inputField.value = '';
		serverError(error.message);
	}//try end 
	return isError;
}


async function sendInput() {
	var llm = document.getElementById('select_llm').value;
	var temp = parseFloat(document.getElementById('set_temp').value);
	var namespace = document.getElementById('select_namespace').value;
	var template = document.getElementById('template').value;
	var top_k = document.getElementById('set_top_k').value;
	var embedd_model = document.getElementById('select_embedd_model').value;
	var paper_id = document.getElementById('set_paper_id').value;
	var fix_keyword = document.getElementById('select_fix_keyword').value;
	var answer_per_paper = document.getElementById('select_answer_per_paper').value;
	var chunks_from_one_paper = document.getElementById('select_chunks_from_one_paper').value;
	var rerank = document.getElementById('select_rerank').value;
	var advanced_use = document.getElementById('select_rows_table').value;
	console.log("advanced_use = " + advanced_use);
	let isError = false;

	if (temp < 0 || temp > 2) {
		console.log("sendInput() should return now");
		// return { icon: 'error', title: 'Invalid params', text: 'temperature parameter must be between 0 and 2' };
	} else {
		/* Extract the query and clear input field */
		let inputField = document.getElementById('user-input');
		let userInput = inputField.value;
		inputField.value = '';

		/* Here you should process the user input and generate a response */
		console.log("Calling fastAPI get_answer endpoint from sendInput() " + ++count + " times!");
		var query = userInput.toLowerCase();
		try {
			const end_point = "/stream_answer"
			const params = {
				llm: llm,
				temp: temp,
				namespace: namespace,
				query: query,
				template: template,
				embedd_model: embedd_model,
				paper_id: paper_id,
				answer_per_paper: answer_per_paper,
				chunks_from_one_paper: chunks_from_one_paper,
				fix_keyword: fix_keyword,
				rerank: rerank,
				top_k: top_k
			};
			// console.log('Before openWebSocket()');
			console.log('Before calling openWebSocket() params.query = ' + params.query);
			// openWebSocket(end_point, params);
			// openWebSocket(end_point, params, handleStreamAnswerMessage);
			openWebSocket(end_point, params, setupChatElements, handleStreamAnswerMessage);
			console.log('After openWebSocket()');
		} catch (error) {
			// // Clear the input field
			// inputField.value = '';
			serverError(error.message);
		}//try end 
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

	/* DO Not Delete */
	document.getElementById('file-input').addEventListener('change', function (event) {
		let file = event.target.files[0];
		if (file.type !== 'application/pdf') {
			console.error(file.name, 'is not a .pdf file');
			return;
		}

		let fileReader = new FileReader();
		fileReader.onload = function (event) {
			let typedArray = new Uint8Array(this.result);
			pdfjsLib.getDocument(typedArray).promise.then((pdf) => {
				pdfViewer.setDocument(pdf);

				// set max value of input field
				document.getElementById('go-to-page').max = pdf.numPages;
				document.getElementById('go-to-page').value = 1;
			});
		};
		fileReader.readAsArrayBuffer(file);
	});
	/* DO Not Delete */

	// // Temporarily blocking the file upload
	// document.getElementById('file-upload-label').addEventListener('click', function (e) {
	// 	e.preventDefault();  // Prevent the default behavior of the label
	// 	Swal.fire({
	// 		icon: 'info',
	// 		// title: "In this version pdf upload is disabled",
	// 		title: "PDF upload is disabled in this version due to limited resources on the cloud server hosting the app. To enable this feature, please set up the app locally by following the instructions in the supplementary file (Additional file 2).",
	// 		allowOutsideClick: false
	// 	});
	// });

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
		// var py_script_path = 'pycodes/get_summary.py';

		// Hide action card
		actionCard.style.display = 'none';
		// Hide summary card (in case it's already shown)
		let summaryCard = document.getElementById('summary-card');
		summaryCard.style.display = 'none';

		// Update summary card content
		try {
			const end_point = "/summarize_selected_text_PDF"
			const params = {
				llm: llm,
				text: text
			};
			ws = new WebSocket(wsUrl + end_point);

			ws.onopen = () => {
				console.log("WebSocket connection opened.");
				document.getElementById('summary-content').textContent = "";
				ws.send(JSON.stringify(params));
			};

			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				console.log("data", data);
				const { summary, error, end_summary } = data;
				Swal.close();

				if (error) {
					console.error(error);
					Swal.fire({
						icon: 'error',
						title: "Summarization failed",
						text: error,
						allowOutsideClick: false
					});
					return;
				}

				if (summary) {
					document.getElementById('summary-content').textContent += summary;
					setTimeout(() => {
						summaryCard.style.display = 'block';
					}, 0);
				}

				if (end_summary) {
					console.log("End of Summary");
					ws.close();
				}
			};

			ws.onerror = (error) => {
				console.error("WebSocket error:", error);
				ws.close();
			};

			ws.onclose = () => {
				console.log("WebSocket connection closed.");
			};

		} catch (error) {
			serverError(error.message);
		}//try end 
	});

	// Add an event listener to the close button to hide the summary card when clicked.
	document.getElementById('close-summary').addEventListener('click', () => {
		document.getElementById('summary-card').style.display = 'none';
	});
});
