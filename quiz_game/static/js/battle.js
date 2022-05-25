// phina.js をグローバル領域に展開
phina.globalize();

const SCREEN_X = window.screen.width;
const SCREEN_Y = window.screen.height;
const MAX_STAGE = 3;

var ASSETS = {
    image: {
        'bg': 'static/png/stage_disease.png',
        'player': 'static/png/player_man.png',
        'maru': 'static/png/maru.png',
        'batu': 'static/png/batu.png',
        'enemy_1': 'static/png/enemy_1.png',
        'enemy_2': 'static/png/enemy_2.png',
        'enemy_3': 'static/png/enemy_3.png',
        'enemy_5': 'static/png/enemy_5.png',
        'enemy_6': 'static/png/enemy_6.png',
    }
}


// MainScene クラスを定義
phina.define('MainScene', {
    superClass: 'DisplayScene',
    init: function(option) {
        this.superInit(option);
        // 背景
        Sprite('bg').addChildTo(this).setPosition(this.gridX.center(), this.gridY.center());
        // ユーザーの作成
        this.player = Player().addChildTo(this).setPosition(this.gridX.center() - (SCREEN_X / 4) , this.gridY.center());
        this.enemy_list = [Enemy_1(), Enemy_6(), Enemy_5()]
        this.phase = "init_stage";
        this.max_stage = MAX_STAGE
        this.stage_num = 0;
        this.question_num = 0;
        this.stage_title = null;
        this.dialogue = null;
        this.maru_batu_mark = null;
        this.enemy = null;
        this.time = 0;
        this.timer = new Timer()
        this.answer_list = [];
        this.answer = null;
    },

    /**
     * phase説明
     * init_stage: ステージの最初
     * show_
     * show_enemy: ダイアログ出現 → 敵が出現
     * 
     * question: クイズダイアログが出現し、入力待ち → 入力がされたら次へ
     * answer: 答えと解説を表示
     * result: クイズの正否に応じたアニメーションを行う
     * 
     * delete_enemy: 敵の消滅
     */
    update: function(app) {
        //console.log(this.dialogue)
        this.time += app.deltaTime;
        console.log(this.phase)
        this.setPhase()
    },

    setPhase: function() {
        switch (this.phase) {
            case "init_stage":
                    if (this.stage_title === null) {
                        if (this.stage_num == MAX_STAGE-1){
                            this.stage_title = StageTitle(`BOSS STAGE`).addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                        } else {
                            this.stage_title = StageTitle(`STAGE ${this.stage_num+1}`).addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                        }
                        this.stage_title.setInteractive(true)
                        var self = this;
                        this.stage_title.onclick = function () {
                            self.stage_title.remove();
                            self.stage_title = null;
                            self.phase = 'show_enemy';
                        };
                    };
                break;
            case "show_enemy":
                // ダイアログ出現
                if (this.dialogue === null) {
                    console.log('実行')
                    this.dialogue = BattleDialogue('敵があらわれた').addChildTo(this).setPosition(this.gridX.center(), this.gridY.center() + SCREEN_Y / 4)
                    // タッチ可能にする
                    this.dialogue.setInteractive(true)
                    var self = this;
                    this.dialogue.onclick = function () {
                        self.dialogue.remove();
                        self.dialogue = null
                        if (self.enemy === null) {
                            self.enemy = self.enemy_list[self.stage_num].addChildTo(self).setPosition(self.gridX.center() + (SCREEN_X / 4) , self.gridY.center())
                            self.enemy.alpha = 0;
                            self.enemy.tweener.fadeIn(1000).play();
                            self.phase = "question_title";
                        }
                    console.log(this.dialogue)
                    };
                };
                break;
            case "question_title":
                this.timer.set_timer(this.time, 2000);
                if (this.timer.check_timer(this.time)){
                    if (this.dialogue === null) {
                        this.dialogue = QuizTitle(`第${this.question_num+1}問`).addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                        this.dialogue.setInteractive(true)
                        var self = this;
                        this.dialogue.onclick = function () {
                            self.dialogue.remove();
                            self.dialogue = null;
                            self.phase = 'show_question';
                            console.log(this.phase)
                        };
                    };
                };
                break;
            case "show_question":
                if(this.dialogue === null) {
                    console.log(question_list)
                    this.dialogue = QuizDialogue(question_list[this.question_num]).addChildTo(this).setPosition(this.gridX.center(), this.gridY.span(10))
                };
                this.answer = this.dialogue.returnSelectedChoice()
                if (this.answer !== null){
                    this.dialogue.remove();
                    this.dialogue = null;
                    this.phase = "check_answer";
                };
                break;
            case "check_answer":
                if (this.maru_batu_mark === null) {
                    if (this.answer == question_list[this.question_num]["answer"]){
                        this.maru_batu_mark = Maru().addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                    } else {
                        this.maru_batu_mark = Batu().addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                    };
                    this.maru_batu_mark.alpha = 0;
                    this.maru_batu_mark.tweener.fadeIn(500).play();
                    this.maru_batu_mark.setInteractive(true)
                    var self = this;
                    this.onclick = function () {
                        this.maru_batu_mark.remove();
                        this.maru_batu_mark= null;
                        this.phase = 'show_answer';
                    };
                };
                break
            case "show_answer":
                console.log("show_answer")
                if(this.dialogue === null) {
                    this.dialogue = AnswerDialogue(question_list[this.question_num]).addChildTo(this).setPosition(this.gridX.center(), this.gridY.span(10));
                    this.dialogue.setInteractive(true);
                    var self = this;
                    this.dialogue.onclick = function() {
                        self.dialogue.remove();
                        self.dialogue = null;
                        self.phase = "attack";
                    };
                };
                break;
            case "attack":
                this.timer.set_timer(this.time, 500);
                if (this.timer.check_timer(this.time)){
                    if (this.answer == question_list[this.question_num]["answer"]){
                        this.player.tweener.moveBy(20, 0, 100).moveBy(-20, 0, 100).play()
                        this.enemy.damage(this.player.attack);
                    } else {
                        this.enemy.tweener.moveBy(-20, 0, 100).moveBy(20, 0, 100).play()
                        this.player.damage(this.enemy.attack);
                    };
                    this.question_num += 1;
                    console.log(this.player.life.value)
                    console.log(this.enemy.life.value)
                    if (this.player.life.value <= 0){
                        this.phase = "game_over"; 
                    } else if (this.enemy.life.value <= 0) {
                        this.enemy = null
                        if(this.stage_num == MAX_STAGE-1) {
                            this.phase = "stage_clear";
                        } else {
                            this.phase = "next_stage";
                        };
                    } else {
                        console.log("next_question")
                        this.phase = "question_title";
                    }
                };
                break;
            case "game_over":
                this.timer.set_timer(this.time, 1000);
                if (this.timer.check_timer(this.time)){
                    if (this.dialogue === null){
                        this.dialogue = ScreenDialogue(this.phase).addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                    };
                };
                break;
            case "next_stage":
                this.timer.set_timer(this.time, 1000);
                if (this.timer.check_timer(this.time)){
                    if (this.dialogue === null) {
                        this.dialogue = BattleDialogue('敵をたおした').addChildTo(this).setPosition(this.gridX.center(), this.gridY.center() + SCREEN_Y / 4)
                        this.dialogue.setInteractive(true)
                        var self = this;
                        this.dialogue.onclick = function () {
                            self.dialogue.remove();
                            self.dialogue = null
                            self.stage_num += 1;
                            self.phase = "init_stage";
                        };
                    };
                };
                break;
            case "stage_clear":
                this.timer.set_timer(this.time, 1000);
                if (this.timer.check_timer(this.time)){
                    if (this.dialogue === null){
                        this.dialogue = ScreenDialogue(this.phase).addChildTo(this).setPosition(this.gridX.center(), this.gridY.center())
                    };
                };
                break;

        };
    },
});

/**
 * usage:
 *      this.timer.set_timer(this.time, 処理を遅らせたい秒数);
            if (this.timer.check_timer(this.time)){
                実行したい処理
            };
 */
class Timer {
    constructor() {
        this.status = 'inactive';
    }
    set_timer(start_time, time_delta) {
        if (this.status == 'inactive') {
            this.start_time = start_time;
            this.time_delta = time_delta;
            this.status = 'active';
        };
    };
    check_timer(now_time) {
        if (this.status == 'active')
        this.now_time = now_time;
        if ((this.now_time - this.start_time) > this.time_delta) {
            this.status = 'inactive';
            return true
        };
    };
}


phina.define("Player", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('player', 256, 256); // (asset, width, height) 
        this.attack = 100
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 500,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        this.life.on('empty', function() {
            self.remove()
        });
    },
    damage: function(damage) {
        var ransuu = this.getRansuu(0.8, 1.1)
        this.life.value -= damage*ransuu;
    },
    getRansuu: function(min, max) {
        var ransuu = Math.random()*(max - min) + min;
        return ransuu
    }
})

phina.define("Maru", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('maru', SCREEN_Y, SCREEN_Y); // (asset, width, height) 
    }
})
phina.define("Batu", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('batu', SCREEN_Y, SCREEN_Y); // (asset, width, height) 
    }
})

/*
 * ライフゲージクラス https://qiita.com/alkn203/items/2210c54afbe6c40645fe
 */
phina.define("LifeGauge", {
    // 継承
    superClass: 'Gauge',
    // コンストラクタ
    init: function(param) {
        // 親クラス初期化
        this.superInit({
            width: param.width,
            height: param.height,
            fill: 'red',
            stroke: 'silver',
            gaugeColor: 'limegreen',
            maxValue: param.life,
            value: param.life,
    });
      // 値変化アニメーションの時間
    this.animationTime = 500;
    },
});

phina.define("StageTitle", {
    superClass: "RectangleShape",
    init: function(text) {
        this.superInit({
            width: SCREEN_X,
            height: SCREEN_Y,
            fill: 'rgb(0, 0, 0, 0.4)',
            stride: null
        })
        if (text == "BOSS STAGE"){
            this.labelArea = LabelArea({
                text: text,
                fill: "white",
                stroke: "red",
                width: 80*6,
                height: 80,
                fontSize: 80,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this);
        } else {
            this.labelArea = LabelArea({
                text: text,
                fill: "white",
                stroke: "red",
                width: 80*4,
                height: 80,
                fontSize: 80,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this);
        };
    },
});

phina.define("ScreenDialogue", {
    superClass: "RectangleShape",
    init: function(phase) {
        this.superInit({
            width: SCREEN_X,
            height: SCREEN_Y,
            fill: 'rgb(0, 0, 0, 0.4)',
            stride: null
        })
        if (phase == "game_over"){
            this.labelArea = LabelArea({
                text: "GAME OVER",
                width: 100 * 7,
                height: 100,
                fill: "white",
                stroke: "red",
                fontSize: 100,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this);
            this.button = RectangleShape({
                width: this.width / 4,
                height: 50 ,
                fill: 'rgb(70, 220, 70, 0.5)',
                stride: 'black',
            }).addChildTo(this).setPosition(0, this.height/5);
            this.buttoLabel = LabelArea({
                text: `ホームに戻る`,
                width: 7*28,
                height: 28,
                fontSize: 28,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this.button);
            this.button.setInteractive(true)
            this.button.onclick = function() {
                location.href = "http://127.0.0.1:8000/home"; // ローカルで起動しない場合も対応できるよう修正する
            };
        } else if (phase == "stage_clear"){
            this.labelArea = LabelArea({
                text: "STAGE CLEAR",
                width: 100 * 7,
                height: 100,
                fill: "white",
                stroke: "red",
                fontSize: 100,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this);
            this.button = RectangleShape({
                width: this.width / 4,
                height: 50 ,
                fill: 'rgb(70, 220, 70, 0.5)',
                stride: 'black',
            }).addChildTo(this).setPosition(0, this.height/5);
            this.buttoLabel = LabelArea({
                text: `ホームに戻る`,
                width: 7*28,
                height: 28,
                fontSize: 28,
                fontFamily: 'Yu Mincho',
            }).addChildTo(this.button);
            this.button.setInteractive(true)
            this.button.onclick = function() {
                location.href = "http://127.0.0.1:8000/home"; // ローカルで起動しない場合も対応できるよう修正する
            };
        }
        
    },
});

phina.define("BattleDialogue", {
    superClass: "RectangleShape",
    init: function(text) {
        this.superInit({
            width: SCREEN_X*0.9,
            height: SCREEN_Y / 3,
            fill: 'rgb(210, 210, 210, 0.7)',
            stride: null
        })
        this.labelArea = LabelArea({
            text: text,
            width: this.width*0.95,
            height: this.height*0.9,
            fill: "black",
            stroke: null,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this);;
    },
});


phina.define("QuizTitle", {
    superClass: "RectangleShape",
    init: function(text) {
        this.superInit({
            width: SCREEN_X / 5,
            height: SCREEN_Y / 7,
            fill: 'rgb(210, 210, 210, 0.7)',
            stride: null
        });
        this.labelArea = LabelArea({
            text: text,
            fill: "black",
            width: text.length*40,
            height: 40,
            stroke: null,
            fontSize: 40,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this).setPosition(0, 0);;
    },
    // 文章の幅を算出
    checkWidth: function (text) {
        let half = 0;
        for (let i = 0; i < text.length; i++) {
            // 半角文字か判定
            if (this.checkHalfWidth(text[i])) half++;
        }
        return (text.length - half / 2) * this.labelArea.fontSize;
    },
    //文字の半角判定
    checkHalfWidth: function (value) {
        return !value.match(/[^\x01-\x7E]/) || !value.match(/[^\uFF65-\uFF9F]/);
    }
});


phina.define("QuizDialogue", {
    superClass: "RectangleShape",
    init: function(question) {
        this.superInit({
            width: SCREEN_X*0.9,
            height: SCREEN_Y*2 / 3,
            fill: 'rgb(210, 210, 210, 0.7)',
            stride: "black",
        });
        this.selected_choice = null;
        // 問題文
        this.questionBox = RectangleShape({
            width: this.width*0.9,
            height: this.height * 10 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, -this.height*6/24);
        this.questionLabel = LabelArea({
            text: `問題1\n${question["question"]}`,
            width: this.questionBox.width*0.95,
            height: this.questionBox.height*0.9,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.questionBox);
        // 選択肢1
        this.choiceBox_1 = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, this.height*1/24);
        this.choiceLabel_1 = LabelArea({
            text: `1. ${question["choice_1"]}`,
            width: this.choiceBox_1.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.choiceBox_1);
        this.choiceBox_1.setInteractive(true)
        var self = this;
        this.choiceBox_1.onclick = function(){
            self.selected_choice = "1";
        }
        // 選択肢2
        this.choiceBox_2 = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, this.height*4/24);
        this.choiceLabel_2 = LabelArea({
            text: `2. ${question["choice_2"]}`,
            width: this.choiceBox_2.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.choiceBox_2);
        this.choiceBox_2.setInteractive(true)
        var self = this;
        this.choiceBox_2.onclick = function(){
            self.selected_choice = "2";
        }
        // 選択肢3
        this.choiceBox_3 = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, this.height*7/24);
        this.choiceLabel_3 = LabelArea({
            text: `3. ${question["choice_3"]}`,
            width: this.choiceBox_3.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.choiceBox_3);
        this.choiceBox_3.setInteractive(true)
        var self = this;
        this.choiceBox_3.onclick = function(){
            self.selected_choice = "3";
        }
        // 選択肢4
        this.choiceBox_4 = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, this.height*10/24);
        this.choiceLabel_4 = LabelArea({
            text: `4. ${question["choice_4"]}`,
            width: this.choiceBox_4.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.choiceBox_4);
        this.choiceBox_4.setInteractive(true)
        var self = this;
        this.choiceBox_4.onclick = function(){
            self.selected_choice = "4";
        }
    },
    returnSelectedChoice: function() {
        return this.selected_choice
    }
})

phina.define("AnswerDialogue", {
    superClass: "RectangleShape",
    init: function(question) {
        this.superInit({
            width: SCREEN_X*0.9,
            height: SCREEN_Y*2 / 3,
            fill: 'rgb(210, 210, 210, 0.7)',
            stride: "black",
        });
        this.answer_choice = "choice_"+question["answer"];
        // 問題文
        this.answerBox = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, -this.height*9/24);
        this.answerLabel = LabelArea({
            text: `正解：${question["answer"]}. ${question[this.answer_choice]}`,
            width: this.answerBox.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.answerBox);
        this.explanationBox = RectangleShape({
            width: this.width*0.9,
            height: this.height * 14 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, 0);
        this.explanationLabel = LabelArea({
            text: `解説:\n${question["explanation"]}`,
            width: this.explanationBox.width*0.95,
            height: this.explanationBox.height*0.9,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.explanationBox);
        this.referenceBox = RectangleShape({
            width: this.width*0.9,
            height: this.height * 3 / 24 ,
            fill: 'rgb(255, 255, 255, 0.7)',
            stride: 'black',
        }).addChildTo(this).setPosition(0, this.height*9/24);
        this.referenceLabel = LabelArea({
            text: `参考:${question["reference"]}`,
            width: this.referenceBox.width*0.95,
            height: 28,
            fontSize: 28,
            fontFamily: 'Yu Mincho',
        }).addChildTo(this.referenceBox);
    },
    returnSelectedChoice: function() {
        return this.selected_choice
    }
})

// メイン処理
phina.main(function() {
    // アプリケーション生成
    var app = GameApp({
        startLabel: 'main', // メインシーンから開始する
        width: SCREEN_X,
        height: SCREEN_Y,
        assets: ASSETS,
    });
    // アプリケーション実行
    app.run();
});