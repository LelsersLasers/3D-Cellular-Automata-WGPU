let paused = false;
let crossSection = false;
let resetFlag = false;

const baseKey = "3D_Cellular_Automata_";

document.querySelectorAll('input[type=radio][name="draw_mode"]').forEach(
    (radio) => {
        radio.addEventListener('change', () => {
            let val = document.querySelector('input[type=radio][name="draw_mode"]:checked').value;
            document.getElementById("draw_mode_rust").value = val;
            hideAllColorInput();
            document.getElementById(val + "_color").hidden = !document.getElementById(val + "_color").hidden;
        });
    }
);
document.querySelectorAll('input[type=text]').forEach(
    (input) => {
        if (input.id.includes("_color") && !input.id.includes("_color_")) {
            input.addEventListener('input', () => {
                applyColorChange(input.id);
            });
        }
    }
);

function getFromLS(key, otherwise) {
    const val = localStorage.getItem(key);
    if (val === null) {
        return otherwise;
    }
    return val;
}
function setFromLoad(id, otherwise) {
    const val = getFromLS(baseKey + id, otherwise);
    document.getElementById(id).value = val;
}

function setToLS(key, value) {
    localStorage.setItem(key, value);
}
function setFromSave(id) {
    const val = document.getElementById(id).value;
    setToLS(baseKey + id, val);
}

function load() {
    setFromLoad("survival_rule_input", "2,6,9");
    setFromLoad("spawn_rule_input", "4,6,8,9");
    setFromLoad("state_rule_input", "10");

    setFromLoad("survival_rule_rust", "2,6,9");
    setFromLoad("spawn_rule_rust", "4,6,8,9");
    setFromLoad("state_rule_rust", "10");
    // NEIGHBORHOOD

    setFromLoad("cell_bounds_input", "96");
    // DRAW MODE

    setFromLoad("dcd_alive_color", "191, 97, 106");
    setFromLoad("sc_start_color", "255, 20, 20");
    setFromLoad("dc_start_color", "255, 20, 20");
    setFromLoad("dc_end_color", "191, 97, 106");
    setFromLoad("cd_max_color", "50, 235, 130");

    setFromLoad("dcd_alive_color_rust", "191,97,106");
    setFromLoad("sc_start_color_rust", "255,20,20");
    setFromLoad("dc_start_color_rust", "255,20,20");
    setFromLoad("dc_end_color_rust", "191,97,106");
    setFromLoad("cd_max_color_rust", "50,235,130");

    resetCells();
}

function save() {
    setFromSave("survival_rule_input");
    setFromSave("spawn_rule_input");
    setFromSave("state_rule_input");

    setFromSave("survival_rule_rust");
    setFromSave("spawn_rule_rust");
    setFromSave("state_rule_rust");
    // NEIGHBORHOOD

    apply();

    setFromSave("cell_bounds_input");
    // DRAW MODE

    setFromSave("dcd_alive_color");
    setFromSave("sc_start_color");
    setFromSave("dc_start_color");
    setFromSave("dc_end_color");
    setFromSave("cd_max_color");

    setFromSave("dcd_alive_color_rust");
    setFromSave("sc_start_color_rust");
    setFromSave("dc_start_color_rust");
    setFromSave("dc_end_color_rust");
    setFromSave("cd_max_color_rust");
}



function applyColorChange(id) {
    const val = document.getElementById(id).value;
    const valTrimmed = val.replace(/\s/g, '');
    let valSplit = valTrimmed.split(",");
    if (valSplit.length != 3) {
        document.getElementById(id).style.border = "2px solid #BF616A";
        return;
    }
    for (let i = 0; i < valSplit.length; i++) {
        let valSplitInt = parseInt(valSplit[i]);
        if (isNaN(valSplitInt) || valSplitInt < 0 || valSplitInt > 255) {
            document.getElementById(id).style.border = "2px solid #BF616A";
            return;
        }
        valSplit[i] = valSplitInt;
    }
    document.getElementById(id + "_rust").value = valSplit.join(",");
    document.getElementById(id).style.border = "none";
}
function hideAllColorInput() {
    document.getElementById("DualColorDying_color").hidden = true;
    document.getElementById("SingleColor_color").hidden = true;
    document.getElementById("DualColor_color").hidden = true;
    document.getElementById("RGB_color").hidden = true;
    document.getElementById("CenterDist_color").hidden = true;
}

function togglePause() {
    paused = !paused;
    if (paused) {
        document.getElementById("pauseButton").innerHTML = "Resume";
    } else {
        document.getElementById("pauseButton").innerHTML = "Pause";
    }
    document.getElementById("paused_rust").value = paused;
}
function toggleCrossSection() {
    crossSection = !crossSection;
    if (crossSection) {
        document.getElementById("crossSectionButton").innerHTML = "Toggle cross section [on]";
    } else {
        document.getElementById("crossSectionButton").innerHTML = "Toggle cross section [off]";
    }
    document.getElementById("cross_section_rust").value = crossSection;
}
function resetCells() {
    resetFlag = !resetFlag;
    document.getElementById("reset_cells_rust").value = resetFlag;
}
function applySurvivalRule() {
    const survivalValue = document.getElementById("survival_rule_input").value;
    const survivalValueTrimmed = survivalValue.replace(/\s/g, '');
    let survivalValueSplit = survivalValueTrimmed.split(",");
    for (let i = 0; i < survivalValueSplit.length; i++) {
        let survivalValueSplitInt = parseInt(survivalValueSplit[i]);
        if (isNaN(survivalValueSplitInt) || survivalValueSplitInt < 0) {
            document.getElementById("survival_rule_input").style.border = "2px solid #BF616A";
            alert("Survival must be a list of positive integers seperated by commas (ex: '2,6,9')");
            return;
        }
        survivalValueSplit[i] = survivalValueSplitInt;
    }
    survivalValueSplit.sort();
    survivalValueStr = survivalValueSplit.join(",");
    document.getElementById("survival_rule_rust").value = survivalValueStr;
    document.getElementById("survival_rule_input").value = survivalValueStr;
    document.getElementById("survival_rule_input").style.border = "none";
}
function applySpawnRule() {
    const spawnValue = document.getElementById("spawn_rule_input").value;
    const spawnValueTrimmed = spawnValue.replace(/\s/g, '');
    let spawnValueSplit = spawnValueTrimmed.split(",");
    for (let i = 0; i < spawnValueSplit.length; i++) {
        const spawnValueSplitInt = parseInt(spawnValueSplit[i]);
        if (isNaN(spawnValueSplitInt) || spawnValueSplitInt < 0) {
            document.getElementById("spawn_rule_input").style.border = "2px solid #BF616A";
            alert("Spawn must be a list of positive integers seperated by commas (ex: '4,6,8,9')");
            return;
        }
        spawnValueSplit[i] = spawnValueSplitInt;
    }
    spawnValueSplit.sort();
    spawnValueStr = spawnValueSplit.join(",");
    document.getElementById("spawn_rule_rust").value = spawnValueStr;
    document.getElementById("spawn_rule_input").value = spawnValueStr;
    document.getElementById("spawn_rule_input").style.border = "none";
}
function applyStateRule() {
    const stateValue = document.getElementById("state_rule_input").value;
    const stateValueInt = parseInt(stateValue);
    if (isNaN(stateValueInt) || stateValueInt < 0) {
        document.getElementById("state_rule_input").style.border = "2px solid #BF616A";
        alert("State must be a positive integer (ex: '10')");
    }
    else {
        document.getElementById("state_rule_input").value = stateValueInt;
        document.getElementById("state_rule_rust").value = stateValueInt;
        document.getElementById("state_rule_input").style.border = "none";
    }
}
function applyCellBounds() {
    const cellBoundsValue = document.getElementById("cell_bounds_input").value;
    const cellBoundsValueInt = parseInt(cellBoundsValue);
    if (isNaN(cellBoundsValueInt) || cellBoundsValueInt < 0) {
        document.getElementById("cell_bounds_input").style.border = "2px solid #BF616A";
        alert("Cell bounds must be a positive integer (ex: '96')");
    }
    else {
        document.getElementById("cell_bounds_input").value = cellBoundsValueInt;
        document.getElementById("cell_bounds_rust").value = cellBoundsValueInt;
        document.getElementById("cell_bounds_input").style.border = "none";
    }
}
function applyNeighborhoodRule() {
    const radioButtons = document.querySelectorAll('input[name="neigborhood"]');
    let neigborhoodValue;
    for (const radioButton of radioButtons) {
        if (radioButton.checked) {
            neigborhoodValue = radioButton.value;
            break;
        }
    }
    document.getElementById("neighborhood_rule_rust").value = neigborhoodValue;
}
function apply() {
    applySurvivalRule();
    applySpawnRule();
    applyStateRule();
    applyNeighborhoodRule();
    applyCellBounds();
}
function resizeCanvas() {
    const canvas = document.getElementsByTagName("canvas")[0];
    if (canvas) {
        let maxWidth = (window.innerWidth - 50) * 0.75;
        let maxHeight = window.innerHeight - 4;
        let width = Math.min(maxWidth, (maxHeight * 16) / 9);
        let height = Math.min(maxHeight, (maxWidth * 9) / 16);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        console.log("Winwdow size:", window.innerWidth, window.innerHeight);
    }
}