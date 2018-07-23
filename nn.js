//data
var data = [ [3, 1.5,   1], [2, 1,   0], [3, 1,   0], [3.5, .5,   1], [2, .5,   0], [1, 1,   0], [5.5, 1,   1], [4, 1.5,   1]  ];
//the flower we will predict
var mystery_flower = [1.5, 1];

var mystery_flower2 = [4.5, 1];

var mystery_flower3 = [2.5, 1.5];

var mystery_flower4 = [2.6, 2];

var mystery_flower5 = [1, 2];
	
	//the "squashing" function
	function sigmoid(x){
		return ((1)/(1+Math.exp(-x)));
	}
	
	//the ""squashing function's derivative
	function sigmoid_p(x){
		return sigmoid(x) * (1-sigmoid(x));
	}
	
	//begin the training
	function train(){
	
		//set the weights and the bias
		let w1 = Math.random();
		let w2 = Math.random();
		let b = Math.random();
		
		let iterations = 10000;
		//learning rate is good at .1, too low = the slope won't reach zero. too high =  the slope might overshoot zero
		let learning_rate = .1;
		
		for(let i=0; i<iterations; i++){

			//get a random data point
			let ri = Math.floor(Math.random() * data.length);
			let point = data[ri];
			
			//feed forward
			let z = point[0] * w1 + point[1] * w2 + b;
			
			//sigmoid the output
			let pred = sigmoid(z);
			
			let target = point[2];
			
			//get the squared error
			let cost = Math.pow((pred-target), 2);

			//take the derivative of cost,pred,w1,w2,b...which leads to chain rule below

			//yes these are manual derivatives...deal with it			
			let dcost_dpred = 2 * (pred-target);
			
			let dpred_dz = sigmoid_p(z);
			
			let dz_dw1 = point[0];
			let dz_dw2 = point[1];
			let dz_db = 1;
			
			//chain rule
			let dcost_dz = dcost_dpred * dpred_dz;
			
			let dcost_dw1 = dcost_dz * dz_dw1;
			let dcost_dw2 = dcost_dz * dz_dw2;
			let dcost_db = dcost_dz * dz_db;

			//subtract the learning rate to get the new parameters			
			w1 -= learning_rate * dcost_dw1;
			w2 -= learning_rate * dcost_dw2;
			b -= learning_rate * dcost_db;
			
			
		}
		return {
				w1: w1,
			    w2: w2,
			    b: b
				};
	}

	function main(){
			var values = train();

			let z = values.w1 * mystery_flower[0] + values.w2 * mystery_flower[1] + values.b;
			nn_pred = sigmoid(z);



			 let z2 = values.w1 * mystery_flower2[0] + values.w2 * mystery_flower2[1] + values.b;
			 nn_pred2 = sigmoid(z2);

			 let z3 = values.w1 * mystery_flower3[0] + values.w2 * mystery_flower3[1] + values.b;
			 nn_pred3 = sigmoid(z3);

			 let z4 = values.w1 * mystery_flower4[0] + values.w2 * mystery_flower4[1] + values.b;
			 nn_pred4 = sigmoid(z4);

			let z5 = values.w1 * mystery_flower5[0] + values.w2 * mystery_flower5[1] + values.b;
			 nn_pred5 = sigmoid(z5);


			return [values.w1, values.w2, values.b, nn_pred, nn_pred2, nn_pred3, nn_pred4, nn_pred5];
	}
	
	main();
	

