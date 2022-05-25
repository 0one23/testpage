// phina.js をグローバル領域に展開
phina.globalize();

phina.define("Enemy_1", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('enemy_1', 256, 256); // (asset, width, height) 
        this.attack = 100
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 100,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        // ライフが無くなった時の処理
        this.life.on('empty', function() {
            self.remove(); 
        });
    },
    damage: function(damage) {
        var ransu = this.getRansu(0.8, 1.1)
        console.log(ransu)
        this.life.value -= damage*ransu;
    },
    getRansu: function(min, max) {
        var ransu = Math.random()*(max - min) + min;
        return ransu
    }
});

phina.define("Enemy_2", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('enemy_2', 256, 256); // (asset, width, height) 
        this.attack = 70
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 100,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        // ライフが無くなった時の処理
        this.life.on('empty', function() {
            self.remove(); 
        });
    },
    damage: function(damage) {
        var ransu = this.getRansu(0.8, 1.1)
        console.log(ransu)
        this.life.value -= damage*ransu;
    },
    getRansu: function(min, max) {
        var ransu = Math.random()*(max - min) + min;
        return ransu
    }
});

phina.define("Enemy_3", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('enemy_3', 256, 256); // (asset, width, height) 
        this.attack = 50
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 100,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        // ライフが無くなった時の処理
        this.life.on('empty', function() {
            self.remove(); 
        });
    },
    damage: function(damage) {
        var ransu = this.getRansu(0.8, 1.1)
        console.log(ransu)
        this.life.value -= damage*ransu;
    },
    getRansu: function(min, max) {
        var ransu = Math.random()*(max - min) + min;
        return ransu
    }
});

phina.define("Enemy_5", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('enemy_5', 400, 400); // (asset, width, height) 
        this.attack = 300
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 400,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        // ライフが無くなった時の処理
        this.life.on('empty', function() {
            self.remove(); 
        });
    },
    damage: function(damage) {
        var ransu = this.getRansu(0.8, 1.1)
        console.log(ransu)
        this.life.value -= damage*ransu;
    },
    getRansu: function(min, max) {
        var ransu = Math.random()*(max - min) + min;
        return ransu
    }
});

phina.define("Enemy_6", {
    superClass: 'Sprite',
    init: function() {
        this.superInit('enemy_6', 256, 256); // (asset, width, height) 
        this.attack = 120
        this.life = LifeGauge({
            width: this.width * 0.8,
            height: this.height / 10,
            life: 100,
            }).addChildTo(this);
            // ライフゲージの位置
        this.life.y = this.height / 2 + this.life.height;
        var self = this;
        // ライフが無くなった時の処理
        this.life.on('empty', function() {
            self.remove(); 
        });
    },
    damage: function(damage) {
        var ransu = this.getRansu(0.8, 1.1)
        console.log(ransu)
        this.life.value -= damage*ransu;
    },
    getRansu: function(min, max) {
        var ransu = Math.random()*(max - min) + min;
        return ransu
    }
});