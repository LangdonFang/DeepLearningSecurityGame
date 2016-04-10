//----- Global variables -------//
// Charts
var RadarChart, rewardChart, actionChart, errorChart;
// Learning agent
var agent;
// Individual utilties
var DefenderUtilities = [];
var AttackerUtilities = [];

// Full utility table
/*
var Utilities = [[100,0,-700,700],
				[-100,100,100,-200]];
	*/

var Utilities = [[0,0,-1,1,1,-1],
				 [1,-1,0,0,-1,1],
				 [-1,1,1,-3,0,0]];
					
var Interval;
var IntervalTime = 0;

//Defender Solutions for error checking
var Solution = [];

var lastReward = 0;

//----- End of Global variables -------//

function start() {
	// Reinitalize all global variables	
	DefenderUtilities = [];
	AttackerUtilities = [];
	lastReward = 0;
	
	//Solve the orginal problem (checking to see if we learn correctly)
	LPSolver();
	// Creating the learning chart
	learningChart();
	// Creating reward graph
	(rewardChart = new Graph("updating-reward","legend-reward")).create();
	// Creating action graph
	(actionChart = new Graph("updating-action","legend-action")).create();
	// Creating error graph
	(errorChart = new Graph("updating-error","legend-error")).createError();
	
	document.getElementById("utilitesText").innerHTML = printUtilities(Utilities); 
	
  	// create environment
	var env = {};
	// Total Utilities and last reward 
	var numUtil = Utilities.length;
	env.getNumStates = function() { return numUtil*numUtil+2; }
	env.getMaxNumActions = function() { return numUtil; }
	
	// create the agent, yay!
	// agent parameter spec to play with (this gets eval()'d on Agent reset)
	var spec = {}
	spec.update = 'qlearn'; // qlearn | sarsa
	// Checking discount (was .9)
	spec.gamma = .9; // discount factor, [0, 1)
	spec.epsilon = 0.2; // initial epsilon for epsilon-greedy policy, [0, 1)
	spec.alpha = 0.0001; // value function learning rate
	spec.experience_add_every = 1; // number of time steps before we add another experience to replay memory
	spec.experience_size = 1000; // size of experience replay memory
	spec.learning_steps_per_iteration = 20; //20
	spec.tderror_clamp = 1.0; // for robustness
	spec.num_hidden_units = 100; // number of neurons in hidden layer
	agent = new RL.DQNAgent(env, spec); 
	
	for(var i=0; i<Utilities.length; ++i){
		for(var j=0; j<Utilities.length; ++j){
			DefenderUtilities.push(Utilities[i][j*2]);
			AttackerUtilities.push(Utilities[i][j*2+1]);
		}
	}
	
	startInterval();
}

function startInterval(){
	Interval = setInterval(function(){ // start the learning loop
		var enterValues = DefenderUtilities.slice();
		enterValues.push(lastReward);
		enterValues.push(rewardChart.defenderCount);
		var action = Number(agent.actRandom(enterValues));
		var actionProb = agent.amat["w"];
		
		//Updating Action Probability Chart
		changeRadarChart(actionProb);
		document.getElementById("learningtext").innerText = printArray(actionProb); 
		
		//Updating error chart
		var error = Math.abs(math.sum(...actionProb)-math.sum(...Solution));
		for(var i=0; i<actionProb.length; ++i){
			error += Math.abs(actionProb[i]-Solution[i]);
		}
	  	errorChart.changeGraphError(error);
		
		//Get attackers action
		// Check if probabilites are all zero than split evenly
		var sum = math.sum(...actionProb);
		if(sum==0) {
			for(var i=0; i<actionProb.length; ++i)
				actionProb[i] = 1/actionProb.length;
		}
		//var attackAction = Math.floor(Math.random() * (Utilities.length));
		var attackAction = 0;
		var attackMax = 0;
		for(var i=0; i<Utilities.length; ++i){
			var tempMax = 0.0;
			for(var j=0; j<Utilities.length; ++j){
				tempMax += Utilities[j][i*2+1]*actionProb[j];
			}
			if(attackAction == -1 || tempMax > attackMax) {
				attackAction = i;
				attackMax = tempMax;
			} else if(tempMax==attackMax){
				var coinflip = Math.floor(Math.random() * (2));
				if(coinflip==0){
					attackAction = i;
					attackMax = tempMax;
				}
			}
		}
		
		
		var rewardValues = [], actionValues = [];
		//Lets get reward for defender
	  	var reward = Utilities[action][attackAction*2];
	  	// Reward graph defender than attacker
	  	rewardValues.push(reward);
	  	rewardValues.push(Utilities[action][attackAction*2+1]);
	  	rewardChart.changeGraphReward(rewardValues);
	  	// Action graph defender than attacker
	  	actionValues.push((action+1));
	  	actionValues.push((attackAction+1));
	  	actionChart.changeGraphAction(actionValues);
	  	
	  	lastReward = reward;
	  	var OldMax = Math.max(Math.abs(Math.min(...DefenderUtilities)),
	  							Math.abs(Math.max(...DefenderUtilities)));
	  	var OldMin = -OldMax;
	  		OldRange = (OldMax - OldMin), NewValue = -1, NewRange = 0;
	  	var newMax = 1; newMin = -1;
		if (OldRange != 0) {
		    NewRange = (newMax - newMin);  
		    NewValue = (((reward - OldMin) * NewRange) / OldRange) + newMin;
		}
	  	// execute action in environment and get the reward
	  	agent.learn(NewValue); // the agent improves its Q,policy,model, etc. reward is a float
	},  IntervalTime);
}

function LPSolver()
{
	var solutions = [];
	// Create
	for(var i=0; i<Utilities.length; ++i) {
		var constraints = [];
		var maxStr = "";
		var equalOneStr = "";
		// Create the max string and the equals one
		for(var j=0; j<Utilities.length; ++j) {
			if(j!=0 && Utilities[j][i*2]>0) {
				maxStr += "+";
			} 
			if (j!=0) {
				equalOneStr += "+";
			}
			maxStr += Utilities[j][i*2]+"x"+(j+1);
			equalOneStr += "x"+(j+1);
		}
		equalOneStr += "=1";
		constraints.push(equalOneStr);
		
		//Create all the constraints
		for(var j=0; j<Utilities.length; ++j){
			var constraint = "";
			if(i!=j){
				for(var t=0; t<Utilities.length; ++t) {
					var num = (Utilities[t][i*2+1]-Utilities[t][j*2+1]);
					if(t!=0 && num>0) {
						constraint += "+";
					} 
					constraint += num+"x"+(t+1);
				}
				constraint += ">=0";
				constraints.push(constraint);
			}
		}
		
		var results = solver.maximize(maxStr, constraints);
		solutions.push(results);
	}
	
	var maxI = 0;
	var max = solutions[maxI].max;
	for(var i=1; i<results.length; ++i) {
		if(max<solutions[i].max) {
			maxI = i;
			max = solutions[i].max;
		}
	}
	var chartArray = [];
	for(var i=0; i<Utilities.length; ++i){
		chartArray.push(solutions[maxI][String("x"+(i+1))]);
	}
	// Saving the solution for error checking
	Solution = chartArray.slice();
	// Displaying the solution radar
	answerChart(chartArray);
	// Printing the models for the user
	document.getElementById("solutiontext").innerText = printArray(chartArray); 
	document.getElementById("solutionModalText").innerHTML = printSolutions(solutions);
}

function learningChart(){
	var ctx = document.getElementById("learning").getContext("2d");
	var labelNums = [];
	var nums = [];
	for(var i=0; i<Utilities.length; ++i) {
		labelNums.push(String(i+1));
		nums.push(1);
	}
	if(labelNums.length<3) {
		labelNums.push("Null");
		nums.push(0);
	}
	var data = {
	    labels: labelNums,
	    datasets: [
	        {
	            label: "Probabilities",
	            data: nums
	        }
		    ]
	};
	RadarChart = new Chart(ctx).Radar(data, {
	    pointDot: false
	});
}

function answerChart(array){
	var ctx = document.getElementById("solution").getContext("2d");
	var labelNums = [];
	for(var i=0; i<array.length; ++i) {
		labelNums.push(String(i+1));
	}
	if(array.length<3) {
		labelNums.push("Null");
		array.push(0);
	}
	var data = {
	    labels: labelNums,
	    datasets: [
	        {
	            label: "Probabilities",
	            data: array
	        }
		    ]
	};
	RadarChart = new Chart(ctx).Radar(data, {
	    pointDot: false
	});
}

function changeRadarChart(values)
{
	if(values.length<3) {
		values[2]= 0;
	}
	for(var i=0; i<values.length; i++){
		RadarChart.datasets[0].points[i].value = values[i];
	}
	RadarChart.update();
}

function stopInterval()
{
	clearInterval(Interval);
} 

function restart(){
	stopInterval();
	start();
}

function printArray(values)
{
	var str = "";
	for(var i=0; i<values.length; ++i) {
		str += "Target ("+(i+1)+"): "+values[i].toFixed(4)+", ";
	}
	return str;
}

function printUtilities(values)
{
	var str = "<div class=\"table-responsive\"><table class=\"table\"><tr>";
	for(var i=0; i<values.length; ++i){
		str += "<th colspan=\"2\">Target "+(i+1)+"</th>";
	}
	str += "</tr>";
	for(var i=0; i<values.length; ++i){
		str += "<tr>";
		for(var j=0; j<values.length*2; ++j){
			var temp = (j%2==0)?"class=\"info\"":"class=\"danger\"";
			str += "<td "+temp+">"+values[i][j]+"</td>";
		}
		str += "</tr>";
	}
	str += "</table></div>";
	return str;
}

function printSolutions(values)
{
	var str = "<div class=\"table-responsive\"><table class=\"table\"><tr>";
	str += "<th>Max</td>"
	for(var i=0; i<Utilities.length; ++i){
		str += "<th>Target "+(i+1)+"</th>";
	}
	str += "</tr>";
	for(var i=0; i<values.length; ++i){
		str += "<tr>";
		str += "<td>"+values[i].max.toFixed(4)+"</td>";
		for(var j=0; j<Utilities.length; ++j){
			str += "<td>P("+values[i][String("x"+(j+1))].toFixed(4)+")</td>";
		}
		str += "</tr>";
	}
	str += "</table></div>";
	return str;
}

function Graph(name,legend){
	this.name = name;
	this.chart;
	this.latestChartLabel = 0;
	this.legend = legend;
	this.changeGraphCounter = 0;
	this.attackCount = 0;
	this.defenderCount = 0;
	this.lastCount = 0;
	this.slope = 0;
}

Graph.prototype.create = function() {
	var canvas = document.getElementById(this.name),
    ctx = canvas.getContext('2d'),
    startingData = {
      labels: [1],
      datasets: [
          {
          	  label: "Defender",
          	  fillColor: "rgba(220,220,220,0.2)",
              strokeColor: "rgba(220,220,220,1)",
              pointColor: "rgba(220,220,220,1)",
              pointStrokeColor: "#fff",
              data: [this.attackCount]
          },
          {
          	  label: "Attacker",
          	  fillColor: "rgba(151,187,205,0.2)",
              strokeColor: "rgba(151,187,205,1)",
              pointColor: "rgba(151,187,205,1)",
              pointStrokeColor: "#fff",
              data: [this.defenderCount]
          }
      ]
    };
    this.latestChartLabel = startingData.labels[0];
    this.chart = new Chart(ctx).Line(startingData,
    	{legendTemplate : "<ul style=\"list-style-type:none\"><% for (var i=0; i<datasets.length; i++){%>"+
    						"<li><span style=\"background-color:<%=datasets[i].pointColor%>\">"+
    						"<%if(datasets[i].label){%><%=datasets[i].label%><%}%></span>"+
    						"</li><%}%></ul>", animation : false});
    var legendtext = this.chart.generateLegend();
    document.getElementById(this.legend).innerHTML = legendtext;
}

Graph.prototype.createError = function() {
	var canvas = document.getElementById(this.name),
    ctx = canvas.getContext('2d'),
    startingData = {
      labels: [1],
      datasets: [
          {
          	  label: "Defender Error",
          	  fillColor: "rgba(151,187,205,0.2)",
              strokeColor: "rgba(151,187,205,1)",
              pointColor: "rgba(151,187,205,1)",
              pointStrokeColor: "#fff",
              data: [this.attackCount]
          }
      ]
    };
    this.latestChartLabel = startingData.labels[0];
    this.chart = new Chart(ctx).Line(startingData,
    	{legendTemplate : "<ul style=\"list-style-type:none\"><% for (var i=0; i<datasets.length; i++){%>"+
    						"<li><span style=\"background-color:<%=datasets[i].pointColor%>\">"+
    						"<%if(datasets[i].label){%><%=datasets[i].label%><%}%></span>"+
    						"</li><%}%></ul>",animation : false, pointDot : false, showTooltips: false, scaleShowVerticalLines: false});
    var legendtext = this.chart.generateLegend();
    document.getElementById(this.legend).innerHTML = legendtext;
}

Graph.prototype.changeGraphReward = function(values) {
	this.defenderCount += values[0];
	this.attackCount += values[1];
	++this.latestChartLabel;
	// Add numbers for each dataset
	if(this.latestChartLabel%100==0) {
		this.chart.addData([this.defenderCount, this.attackCount], this.latestChartLabel);
		++this.changeGraphCounter;		
	}
	// Remove the first point so we dont just add values forever
	if(this.changeGraphCounter >10 && this.latestChartLabel%100==0)
		this.chart.removeData();
}

Graph.prototype.changeGraphAction = function(values) {
	++this.latestChartLabel;
	if(this.defenderCount==0 || this.attackCount==0 || 
			this.defenderCount!=values[0] || this.attackCount!=values[1]){
		this.defenderCount = values[0];
		this.attackCount = values[1];
		// Add numbers for each dataset
		this.chart.addData([this.defenderCount,this.attackCount], this.latestChartLabel);
		this.changeGraphCounter++;
		// Remove the first point so we dont just add values forever
		if(this.changeGraphCounter>10)
			this.chart.removeData();
	}
}

Graph.prototype.changeGraphError = function(value) {
	++this.latestChartLabel;
	this.defenderCount = value;
	// Add numbers for each dataset
	var yAxisSlope = this.defenderCount-this.lastCount;
	if(this.latestChartLabel < 3000 && 
		(Math.sign(this.slope)!=Math.sign(yAxisSlope) ||
		  this.latestChartLabel%100==0)) {
		this.slope = yAxisSlope;
		var label = "";
		if(this.latestChartLabel%100==0)
			label = this.latestChartLabel;
		this.chart.addData([this.defenderCount], label);
	}
	this.lastCount = this.defenderCount;
}
