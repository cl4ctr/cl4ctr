download from NFM https://github.com/hexiangnan/neural_factorization_machine
Frappe dataset v1.0 http://baltrunas.info/research-menu/frappe

The frappe dataset contains a context-aware app usage log.
It consist of 96203 entries by 957 users for 4082 apps used in various contexts.
(sample 2 negative samples for 1 positive => # of total instances: 288609)

Nonzero u-i pairs: 18842
Context fields:
	#user:  957
	#item:  4082
	#cnt:  1981 (means how many times the app has been used by the user; convert it to 0/1，the target)
	#daytime:  7
	#weekday:  7
	#isweekend:  2
	#homework:  3
	#cost:  2
	#weather:  9
	#country:  80
	#city:  233
Total features: 7363 - 1981 = 5382