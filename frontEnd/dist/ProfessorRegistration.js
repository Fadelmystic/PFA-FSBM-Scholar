// src/ProfessorRegistration.ts
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
// --- Helper Functions (Defined ONCE at the top) ---
function showError(inputId, message) {
    const input = document.getElementById(inputId);
    if (!input)
        return;
    const formGroup = input.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    input.classList.add('invalid');
    input.setAttribute('aria-invalid', 'true');
    if (errorSpan) {
        errorSpan.textContent = message;
        errorSpan.classList.add('visible');
        input.setAttribute('aria-describedby', errorSpan.id || '');
    }
}
function clearError(inputId) {
    const input = document.getElementById(inputId);
    if (!input)
        return;
    const formGroup = input.closest('.form-group');
    const errorSpan = formGroup === null || formGroup === void 0 ? void 0 : formGroup.querySelector('.error-message');
    input.classList.remove('invalid');
    input.removeAttribute('aria-invalid');
    input.removeAttribute('aria-describedby');
    if (errorSpan) {
        errorSpan.textContent = '';
        errorSpan.classList.remove('visible');
    }
}
// --- End Helper Functions ---
const form = document.getElementById("registerForm");
if (form) {
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        console.log("Professor form submitted...");
        const nomInput = document.getElementById("nom");
        const prenomInput = document.getElementById("prenom");
        const emailInput = document.getElementById("email");
        const modulesInput = document.getElementById("modules");
        const passwordInput = document.getElementById("password");
        const confirmPasswordInput = document.getElementById("confirmPassword");
        if (!nomInput || !prenomInput || !emailInput || !modulesInput || !passwordInput || !confirmPasswordInput) {
            console.error("Form elements missing!");
            alert("Erreur de formulaire.");
            return;
        }
        let isValid = true;
        const inputs = [nomInput, prenomInput, emailInput, modulesInput, passwordInput, confirmPasswordInput];
        inputs.forEach(input => input ? clearError(input.id) : null); // Use helpers defined above
        // Validation logic...
        if (!nomInput.value.trim()) {
            showError("nom", "Le nom est requis.");
            isValid = false;
        }
        if (!prenomInput.value.trim()) {
            showError("prenom", "Le prénom est requis.");
            isValid = false;
        }
        if (!emailInput.value.trim()) {
            showError("email", 'L\'email est requis.');
            isValid = false;
        }
        else if (!/\S+@\S+\.\S+/.test(emailInput.value)) {
            showError("email", 'Veuillez entrer une adresse email valide.');
            isValid = false;
        }
        if (!modulesInput.value.trim()) {
            showError("modules", "Veuillez indiquer le(s) module(s).");
            isValid = false;
        }
        const passwordValue = passwordInput.value;
        if (!passwordValue) {
            showError("password", "Le mot de passe est requis.");
            isValid = false;
        }
        else if (passwordValue.length < 8) {
            showError("password", "Le mot de passe doit contenir au moins 8 caractères.");
            isValid = false;
        }
        const confirmPasswordValue = confirmPasswordInput.value;
        if (!confirmPasswordValue) {
            showError("confirmPassword", "Veuillez confirmer le mot de passe.");
            isValid = false;
        }
        else if (passwordValue && passwordValue.length >= 8 && passwordValue !== confirmPasswordValue) {
            showError("confirmPassword", "Les mots de passe ne correspondent pas.");
            showError("password", "Les mots de passe ne correspondent pas.");
            isValid = false;
        }
        if (!isValid) {
            console.log("Professor validation failed.");
            const firstInvalid = form.querySelector('.invalid');
            firstInvalid === null || firstInvalid === void 0 ? void 0 : firstInvalid.focus();
            return;
        }
        const user = {
            nom: nomInput.value,
            prenom: prenomInput.value,
            email: emailInput.value,
            password: passwordInput.value,
            role: 'PROFESSOR',
            confirmPassword: confirmPasswordValue,
            isActive: false,
            isVerified: false,
            verificationCode: null,
            verificationCodeExpiry: null
        };
        const response = registerProfessor(user);
        console.log(response);
    });
    const registerProfessor = (user) => __awaiter(void 0, void 0, void 0, function* () {
        try {
            const response = yield fetch("http://localhost:8443/api/auth/register/professor", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(user)
            });
            if (!response.ok) {
                const error = yield response.json();
                throw new Error(error.message || "Erreur d'inscription");
            }
            const data = yield response.json();
            console.log(data);
        }
        catch (err) {
            console.error("Erreur d'inscription :", err.message);
            alert("Erreur : " + err.message);
        }
    });
}
else {
    console.error("Formulaire Professeur (registerForm) introuvable !");
}
export {};
